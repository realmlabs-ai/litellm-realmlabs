"""
RealmLabs MLS guardrail integration for LiteLLM.

Calls the RealmLabs MLS guardrail endpoint, which returns classifier probe
verdicts and PII spans for a conversation from a single forward pass. On the
request side this guardrail:

  1. blocks when the ``hazard_prompt`` probe scores above ``hazard_threshold``
  2. otherwise masks any detected PII, or blocks if ``pii_mask`` is False

Only ``pre_call`` is supported today; response-side scanning is not wired up
yet, so model-generated PII is not masked.

The guardrail is stateless: MLS can accumulate conversation history itself
under an ``MLS_turn_id``, but LiteLLM already has the full conversation on
every request, so it is sent in full and no turn id is tracked. That keeps the
hook idempotent under retries and immune to MLS process restarts.
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
        return [GuardrailEventHooks.pre_call]

    def _hazard_score(self, response: RealmLabsGuardrailResponse) -> float | None:
        """Score of the hazard_prompt probe, or None if MLS did not run it."""
        for result in response.get("results") or []:
            if result.get("probe") == _HAZARD_PROBE:
                return result.get("prob")
        return None

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
        structured_messages: Final = inputs.get("structured_messages") or []

        if structured_messages:
            messages: list[dict[str, Any]] = [dict(message) for message in structured_messages]
        elif texts:
            messages = [{"role": "user", "content": text} for text in texts]
        else:
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

        # 1. Hazard: block before masking, so a hazardous prompt is rejected
        #    outright rather than masked and forwarded to the model.
        hazard_score: Final = self._hazard_score(result)
        if hazard_score is not None and hazard_score > self.hazard_threshold:
            verbose_proxy_logger.warning(
                "RealmLabs MLS blocked request: %s=%s > %s",
                _HAZARD_PROBE,
                hazard_score,
                self.hazard_threshold,
            )
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=(
                    f"Blocked by RealmLabs {_HAZARD_PROBE} probe: "
                    f"score={hazard_score} exceeds threshold={self.hazard_threshold}"
                ),
            )

        # 2. PII
        spans: Final = result.get("pii_spans") or []
        if not spans:
            return inputs

        if not self.pii_mask:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Blocked by RealmLabs: PII detected in the request ({self._span_types(spans)})",
            )

        masked_texts: Final = [self._mask_pii_in_text(text, spans) for text in texts]
        if masked_texts != texts:
            verbose_proxy_logger.debug(
                "RealmLabs MLS masked PII types: %s",
                self._span_types(spans),
            )
            inputs["texts"] = masked_texts

        return inputs
