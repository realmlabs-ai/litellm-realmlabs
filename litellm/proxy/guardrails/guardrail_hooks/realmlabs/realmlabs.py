"""
RealmLabs MLS guardrail integration for LiteLLM.

Calls the RealmLabs MLS guardrail endpoint, which returns classifier probe
verdicts and PII spans for a conversation from a single forward pass.

  pre_call   the request. Blocks when the ``hazard_prompt`` probe scores above
             ``hazard_threshold``, otherwise masks any detected PII (or blocks,
             when ``pii_mask`` is False) before the model sees it.
  post_call  the model's reply, analysed in the context of the conversation
             that produced it. PII is masked before the caller sees it.

The MLS endpoint is stateless and stores nothing: every call must carry the
whole conversation it wants analysed, with the system prompt included as a
leading system message because the chat template injects none of its own.
Both hooks therefore send the full conversation - pre_call the messages as
submitted, post_call those same messages plus the model's reply as a trailing
assistant turn. Nothing is threaded between the two hooks, which keeps them
idempotent under retries.

``hazard_prompt`` scores the user turn (``expected_role`` is ``user``), so it
is only enforced on the request side, and only when MLS does not report a
``role_mismatch``.
"""

import os
from typing import TYPE_CHECKING, Any, Final, Literal, Optional

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.realmlabs import (
    RealmLabsGuardrailResponse,
    RealmLabsPIISpan,
    RealmLabsProbeResult,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )

_DEFAULT_API_BASE: Final = "https://mls.realmlabs.ai"
_GUARDRAIL_ENDPOINT: Final = "/litellm/guardrail"
_HAZARD_PROBE: Final = "hazard_prompt"
_DEFAULT_HAZARD_THRESHOLD: Final = 0.703
_DEFAULT_TIMEOUT: Final = 15.0


class RealmLabsMissingCredentials(Exception):
    pass


class RealmLabsGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        probes: list[str] | str | None = None,
        hazard_threshold: float | None = None,
        pii: bool | None = None,
        pii_mask: bool | None = None,
        block_on_error: bool | None = None,
        enable_thinking: bool | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key or os.environ.get("REALMLABS_API_KEY")
        if not self.api_key:
            raise RealmLabsMissingCredentials(
                "RealmLabs API key is required. Set REALMLABS_API_KEY in the environment "
                "or pass api_key in the guardrail config."
            )

        self.api_base = (api_base or os.environ.get("REALMLABS_API_BASE") or _DEFAULT_API_BASE).rstrip("/")

        # Only hazard_prompt's score is enforced, so that is all we ask for by
        # default - MLS bills a forward pass either way, but a narrower probe
        # list keeps the response small.
        self.probes: list[str] | str = probes if probes is not None else [_HAZARD_PROBE]
        self.hazard_threshold = hazard_threshold if hazard_threshold is not None else _DEFAULT_HAZARD_THRESHOLD
        self.pii = True if pii is None else pii
        self.pii_mask = True if pii_mask is None else pii_mask
        self.block_on_error = False if block_on_error is None else block_on_error
        self.enable_thinking = False if enable_thinking is None else enable_thinking
        self.timeout = timeout if timeout is not None else _DEFAULT_TIMEOUT

        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )

        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))

        super().__init__(**kwargs)

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.realmlabs import (
            RealmLabsGuardrailConfigModel,
        )

        return RealmLabsGuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]

    @staticmethod
    def _hazard_result(response: RealmLabsGuardrailResponse) -> RealmLabsProbeResult | None:
        """The hazard_prompt verdict, or None if MLS did not run that probe."""
        for result in response.get("results") or []:
            if result.get("probe") == _HAZARD_PROBE:
                return result
        return None

    @staticmethod
    def _message_text(content: Any) -> str | None:
        """Plain text of a message, flattening OpenAI multimodal content parts."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: Final = [
                part.get("text")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
            ]
            if parts:
                return "\n".join(parts)
        return None

    def _conversation_messages(self, request_data: dict) -> list[dict[str, Any]]:
        """The conversation as submitted, from the request LiteLLM is serving.

        Available on both hooks - request_data is the same request dict either
        side of the call - which is what lets post_call analyse the reply in
        context without any state carried over from pre_call.
        """
        messages: Final[list[dict[str, Any]]] = []
        for message in request_data.get("messages") or []:
            if not isinstance(message, dict):
                continue
            text = self._message_text(message.get("content"))
            role = message.get("role")
            if text and role:
                messages.append({"role": str(role), "content": text})
        return messages

    def _mask_pii_in_text(self, text: str, spans: list[RealmLabsPIISpan]) -> str:
        """Rewrite every detected span in ``text`` as its type in brackets.

        MLS reports start/end against its rendered view of the whole
        conversation, so those offsets do not address this single message and
        are ignored - each span's literal ``text`` is located here instead.
        Repeats are masked one at a time from the front, because masking
        shifts everything after it.
        """
        masked: str = text
        for span in spans:
            found: Final = span.get("text")
            entity_type: Final = span.get("type")
            if not found or not entity_type:
                continue
            mask: Final = f"[{entity_type}]"
            while True:
                index: Final[int] = masked.find(found)
                if index == -1:
                    break
                masked = self.mask_content_in_string(
                    masked,
                    mask,
                    index,
                    index + len(found),
                )
        return masked

    @staticmethod
    def _span_types(spans: list[RealmLabsPIISpan]) -> str:
        types: Final[list[str]] = []
        for span in spans:
            entity_type = span.get("type")
            if entity_type and entity_type not in types:
                types.append(entity_type)
        return ", ".join(types) if types else "unknown"

    async def _call_mls(self, messages: list[dict[str, Any]]) -> RealmLabsGuardrailResponse:
        # NB: no "model" field. MLS's request schema types `model` as an object
        # describing the MLS-hosted classifier, not the downstream LLM, so
        # forwarding LiteLLM's model name makes it reject the call with a 422.
        payload: Final[dict[str, Any]] = {
            "messages": messages,
            "probes": self.probes,
            "pii": self.pii,
            "enable_thinking": self.enable_thinking,
        }

        endpoint: Final = f"{self.api_base}{_GUARDRAIL_ENDPOINT}"
        verbose_proxy_logger.debug(
            "RealmLabs MLS: %s msgs=%d probes=%s pii=%s",
            endpoint,
            len(messages),
            self.probes,
            self.pii,
        )

        response: Final = await self.async_handler.post(
            url=endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        texts: Final = inputs.get("texts") or []
        is_request: Final = input_type == "request"

        # MLS is stateless, so every call carries the whole conversation. On the
        # response side that is the same conversation plus the model's reply, so
        # the reply is scored in the context that produced it.
        conversation: Final = self._conversation_messages(request_data)
        if is_request:
            structured_messages: Final = inputs.get("structured_messages") or []
            if structured_messages:
                messages: list[dict[str, Any]] = [dict(message) for message in structured_messages]
            elif conversation:
                messages = list(conversation)
            else:
                messages = [{"role": "user", "content": text} for text in texts]
        else:
            messages = list(conversation) + [{"role": "assistant", "content": text} for text in texts]

        if not messages:
            return inputs

        try:
            result: RealmLabsGuardrailResponse = await self._call_mls(messages)
        except Exception as exc:
            verbose_proxy_logger.error("RealmLabs MLS error: %s", str(exc))
            if self.block_on_error:
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"RealmLabs MLS unreachable (block_on_error=True): {exc}",
                ) from exc
            return inputs

        # 1. Hazard. Request side only: the probe scores the user turn, so on a
        #    response the focal turn is the assistant's and MLS flags a
        #    role_mismatch. Checked before masking, so a hazardous prompt is
        #    rejected outright rather than masked and forwarded.
        hazard: Final = self._hazard_result(result)
        if is_request and hazard is not None and not hazard.get("role_mismatch"):
            score: Final = hazard.get("prob")
            if score is not None and score > self.hazard_threshold:
                verbose_proxy_logger.warning(
                    "RealmLabs MLS blocked request: %s=%s > %s",
                    _HAZARD_PROBE,
                    score,
                    self.hazard_threshold,
                )
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=(
                        f"Blocked by RealmLabs {_HAZARD_PROBE} probe: "
                        f"score={score} exceeds threshold={self.hazard_threshold}"
                    ),
                )

        # 2. PII, both sides.
        spans: Final = result.get("pii_spans") or []
        if not spans:
            return inputs

        if not self.pii_mask:
            where: Final = "request" if is_request else "response"
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Blocked by RealmLabs: PII detected in the {where} ({self._span_types(spans)})",
            )

        # Spans cover the whole conversation, so those belonging to earlier
        # turns simply will not be found in the texts this hook owns.
        masked_texts: Final = [self._mask_pii_in_text(text, spans) for text in texts]
        if masked_texts != texts:
            verbose_proxy_logger.debug(
                "RealmLabs MLS masked PII types in the %s: %s",
                "request" if is_request else "response",
                self._span_types(spans),
            )
            inputs["texts"] = masked_texts

        return inputs
