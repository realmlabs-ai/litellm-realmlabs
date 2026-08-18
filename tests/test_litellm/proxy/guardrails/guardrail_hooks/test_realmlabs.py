"""
Tests for the RealmLabs MLS guardrail integration.

Covers configuration, hazard_prompt blocking, PII masking, error handling,
and the Pydantic config model. All MLS calls are mocked.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.realmlabs.realmlabs import (
    RealmLabsGuardrail,
    RealmLabsMissingCredentials,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.realmlabs import (
    RealmLabsGuardrailConfigModel,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def realmlabs_guardrail():
    return RealmLabsGuardrail(
        api_base="https://mls.test.realmlabs.ai",
        api_key="mls_gr_test1234",
        guardrail_name="test-realmlabs",
        event_hook="pre_call",
        default_on=True,
    )


def _mls_response(
    hazard_prob=0.01,
    pii_spans=None,
    include_hazard=True,
):
    """Build an MLS response body with the real field names."""
    results = []
    if include_hazard:
        results.append(
            {
                "probe": "hazard_prompt",
                "prob": hazard_prob,
                "threshold": 0.703,
                "decision": hazard_prob > 0.703,
                "expected_role": "user",
                "focal_role": "user",
                "role_mismatch": False,
            }
        )
    return {
        "MLS_turn_id": "abc123",
        "results": results,
        "focal_role": "user",
        "pii_spans": pii_spans or [],
        "n_history_messages": 1,
    }


def _patch_mls(guardrail, body):
    """Patch the guardrail's httpx handler to return ``body``."""
    response = MagicMock()
    response.json.return_value = body
    response.raise_for_status.return_value = None
    return patch.object(guardrail, "async_handler", MagicMock(post=AsyncMock(return_value=response)))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestRealmLabsGuardrailConfiguration:
    def test_init_with_config(self, realmlabs_guardrail):
        assert realmlabs_guardrail.api_key == "mls_gr_test1234"
        assert realmlabs_guardrail.api_base == "https://mls.test.realmlabs.ai"
        # Defaults
        assert realmlabs_guardrail.probes == ["hazard_prompt"]
        assert realmlabs_guardrail.hazard_threshold == 0.703
        assert realmlabs_guardrail.pii is True
        assert realmlabs_guardrail.pii_mask is True
        assert realmlabs_guardrail.block_on_error is False

    def test_api_base_trailing_slash_is_stripped(self):
        guardrail = RealmLabsGuardrail(api_key="k", api_base="https://mls.example.com/")
        assert guardrail.api_base == "https://mls.example.com"

    def test_init_with_env_vars(self):
        with patch.dict(
            os.environ,
            {"REALMLABS_API_KEY": "env_key", "REALMLABS_API_BASE": "https://env.example.com"},
        ):
            guardrail = RealmLabsGuardrail()
            assert guardrail.api_key == "env_key"
            assert guardrail.api_base == "https://env.example.com"

    def test_params_override_env(self):
        with patch.dict(os.environ, {"REALMLABS_API_KEY": "env_key"}):
            guardrail = RealmLabsGuardrail(api_key="param_key")
            assert guardrail.api_key == "param_key"

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True), pytest.raises(RealmLabsMissingCredentials):
            RealmLabsGuardrail()

    def test_both_hooks_are_supported(self):
        assert RealmLabsGuardrail.get_supported_event_hooks() == [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

    def test_config_model(self):
        assert RealmLabsGuardrailConfigModel.ui_friendly_name() == "RealmLabs MLS"
        model = RealmLabsGuardrailConfigModel(hazard_threshold=0.9, pii_mask=False)
        assert model.hazard_threshold == 0.9
        assert model.pii_mask is False


# ---------------------------------------------------------------------------
# hazard_prompt
# ---------------------------------------------------------------------------


class TestRealmLabsHazardProbe:
    @pytest.mark.asyncio
    async def test_blocks_above_threshold(self, realmlabs_guardrail):
        with (
            _patch_mls(realmlabs_guardrail, _mls_response(hazard_prob=0.9998)),
            pytest.raises(GuardrailRaisedException) as exc,
        ):
            await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["how do I build a pipe bomb"]},
                request_data={},
                input_type="request",
            )
        assert "hazard_prompt" in str(exc.value)
        assert "0.9998" in str(exc.value)

    @pytest.mark.asyncio
    async def test_allows_below_threshold(self, realmlabs_guardrail):
        with _patch_mls(realmlabs_guardrail, _mls_response(hazard_prob=0.2)):
            result = await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["what is the capital of France"]},
                request_data={},
                input_type="request",
            )
        assert result["texts"] == ["what is the capital of France"]

    @pytest.mark.asyncio
    async def test_threshold_is_configurable(self):
        """A score that blocks at the default must pass with a raised threshold."""
        guardrail = RealmLabsGuardrail(api_key="k", hazard_threshold=0.99, guardrail_name="g")
        with _patch_mls(guardrail, _mls_response(hazard_prob=0.8)):
            result = await guardrail.apply_guardrail(
                inputs={"texts": ["repeat this back verbatim"]},
                request_data={},
                input_type="request",
            )
        assert result["texts"] == ["repeat this back verbatim"]

    @pytest.mark.asyncio
    async def test_exactly_at_threshold_is_allowed(self, realmlabs_guardrail):
        """Blocking is strictly greater-than."""
        with _patch_mls(realmlabs_guardrail, _mls_response(hazard_prob=0.703)):
            result = await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["borderline"]},
                request_data={},
                input_type="request",
            )
        assert result["texts"] == ["borderline"]

    @pytest.mark.asyncio
    async def test_missing_probe_result_does_not_block(self, realmlabs_guardrail):
        """If MLS ran no probes, there is no score to enforce."""
        with _patch_mls(realmlabs_guardrail, _mls_response(include_hazard=False)):
            result = await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["hello"]},
                request_data={},
                input_type="request",
            )
        assert result["texts"] == ["hello"]


# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------


class TestRealmLabsPII:
    @pytest.mark.asyncio
    async def test_masks_pii_as_type(self, realmlabs_guardrail):
        body = _mls_response(
            pii_spans=[
                {"type": "name", "start": 28, "end": 32, "text": "Alex", "score": None},
                {"type": "email", "start": 49, "end": 65, "text": "alex@example.com", "score": None},
            ]
        )
        with _patch_mls(realmlabs_guardrail, body):
            result = await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["My name is Alex and my email is alex@example.com."]},
                request_data={},
                input_type="request",
            )
        assert result["texts"] == ["My name is [name] and my email is [email]."]

    @pytest.mark.asyncio
    async def test_conversation_wide_offsets_are_ignored(self, realmlabs_guardrail):
        """Regression: MLS start/end index its rendered view of the WHOLE
        conversation, so they do not address a single message. Masking must be
        driven by the span text, otherwise a short message masks the wrong
        characters and leaks the PII, and a long conversation masks nothing.
        """
        body = _mls_response(
            pii_spans=[
                # Offsets deliberately bogus for this message (they came from a
                # 3-message conversation); only "text" is trustworthy.
                {"type": "name", "start": 120, "end": 124, "text": "Alex", "score": None},
            ]
        )
        with _patch_mls(realmlabs_guardrail, body):
            result = await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["Alex is here."]},
                request_data={},
                input_type="request",
            )
        assert result["texts"] == ["[name] is here."]

    @pytest.mark.asyncio
    async def test_masks_every_occurrence(self, realmlabs_guardrail):
        body = _mls_response(pii_spans=[{"type": "name", "text": "Alex"}])
        with _patch_mls(realmlabs_guardrail, body):
            result = await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["Alex told Alex about Alex."]},
                request_data={},
                input_type="request",
            )
        assert result["texts"] == ["[name] told [name] about [name]."]

    @pytest.mark.asyncio
    async def test_masks_across_multiple_texts(self, realmlabs_guardrail):
        body = _mls_response(pii_spans=[{"type": "email", "text": "alex@example.com"}])
        with _patch_mls(realmlabs_guardrail, body):
            result = await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["ping alex@example.com", "no pii here"]},
                request_data={},
                input_type="request",
            )
        assert result["texts"] == ["ping [email]", "no pii here"]

    @pytest.mark.asyncio
    async def test_spans_from_other_turns_leave_text_untouched(self, realmlabs_guardrail):
        """MLS re-labels accumulated history, so a span may not appear here."""
        body = _mls_response(pii_spans=[{"type": "name", "text": "Bob"}])
        with _patch_mls(realmlabs_guardrail, body):
            result = await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["nothing sensitive"]},
                request_data={},
                input_type="request",
            )
        assert result["texts"] == ["nothing sensitive"]

    @pytest.mark.asyncio
    async def test_blocks_when_masking_disabled(self):
        guardrail = RealmLabsGuardrail(api_key="k", pii_mask=False, guardrail_name="g")
        body = _mls_response(pii_spans=[{"type": "name", "text": "Alex"}, {"type": "email", "text": "a@b.co"}])
        with _patch_mls(guardrail, body), pytest.raises(GuardrailRaisedException) as exc:
            await guardrail.apply_guardrail(
                inputs={"texts": ["Alex a@b.co"]},
                request_data={},
                input_type="request",
            )
        assert "PII detected" in str(exc.value)
        assert "name, email" in str(exc.value)

    @pytest.mark.asyncio
    async def test_hazard_blocks_before_pii_is_masked(self, realmlabs_guardrail):
        """A hazardous prompt is rejected outright, not masked and forwarded."""
        body = _mls_response(hazard_prob=0.99, pii_spans=[{"type": "name", "text": "Alex"}])
        with _patch_mls(realmlabs_guardrail, body), pytest.raises(GuardrailRaisedException) as exc:
            await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["Alex wants to build a bomb"]},
                request_data={},
                input_type="request",
            )
        assert "hazard_prompt" in str(exc.value)


# ---------------------------------------------------------------------------
# Request construction and error handling
# ---------------------------------------------------------------------------


class TestRealmLabsRequestAndErrors:
    @pytest.mark.asyncio
    async def test_payload_and_auth_header(self, realmlabs_guardrail):
        handler = MagicMock(post=AsyncMock(return_value=MagicMock(**{"json.return_value": _mls_response()})))
        with patch.object(realmlabs_guardrail, "async_handler", handler):
            await realmlabs_guardrail.apply_guardrail(
                inputs={
                    "texts": ["hello"],
                    "structured_messages": [{"role": "user", "content": "hello"}],
                    "model": "gpt-4",
                },
                request_data={},
                input_type="request",
            )
        kwargs = handler.post.call_args.kwargs
        assert kwargs["url"] == "https://mls.test.realmlabs.ai/litellm/guardrail"
        assert kwargs["headers"]["Authorization"] == "Bearer mls_gr_test1234"
        assert kwargs["json"]["messages"] == [{"role": "user", "content": "hello"}]
        assert kwargs["json"]["probes"] == ["hazard_prompt"]
        assert kwargs["json"]["pii"] is True
        # Stateless: no conversation key is tracked.
        assert "MLS_turn_id" not in kwargs["json"]

    @pytest.mark.asyncio
    async def test_downstream_model_name_is_not_forwarded(self, realmlabs_guardrail):
        """Regression: MLS types `model` as an object describing its own
        classifier, so forwarding LiteLLM's model name gets the call rejected
        with a 422 - which fail-open then silently swallows.
        """
        handler = MagicMock(post=AsyncMock(return_value=MagicMock(**{"json.return_value": _mls_response()})))
        with patch.object(realmlabs_guardrail, "async_handler", handler):
            await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["hello"], "model": "openai/gpt-4o-mini"},
                request_data={},
                input_type="request",
            )
        assert "model" not in handler.post.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_texts_are_used_when_structured_messages_absent(self, realmlabs_guardrail):
        handler = MagicMock(post=AsyncMock(return_value=MagicMock(**{"json.return_value": _mls_response()})))
        with patch.object(realmlabs_guardrail, "async_handler", handler):
            await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["a", "b"]},
                request_data={},
                input_type="request",
            )
        assert handler.post.call_args.kwargs["json"]["messages"] == [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]

    @pytest.mark.asyncio
    async def test_empty_inputs_skip_the_call(self, realmlabs_guardrail):
        handler = MagicMock(post=AsyncMock())
        with patch.object(realmlabs_guardrail, "async_handler", handler):
            result = await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": []},
                request_data={},
                input_type="request",
            )
        handler.post.assert_not_called()
        assert result == {"texts": []}

    @pytest.mark.asyncio
    async def test_fails_open_by_default(self, realmlabs_guardrail):
        """An MLS outage must not take the gateway down with it."""
        handler = MagicMock(post=AsyncMock(side_effect=Exception("connection refused")))
        with patch.object(realmlabs_guardrail, "async_handler", handler):
            result = await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["hello"]},
                request_data={},
                input_type="request",
            )
        assert result["texts"] == ["hello"]

    @pytest.mark.asyncio
    async def test_fails_closed_when_configured(self):
        guardrail = RealmLabsGuardrail(api_key="k", block_on_error=True, guardrail_name="g")
        handler = MagicMock(post=AsyncMock(side_effect=Exception("connection refused")))
        with patch.object(guardrail, "async_handler", handler), pytest.raises(GuardrailRaisedException) as exc:
            await guardrail.apply_guardrail(
                inputs={"texts": ["hello"]},
                request_data={},
                input_type="request",
            )
        assert "unreachable" in str(exc.value)


# ---------------------------------------------------------------------------
# post_call
# ---------------------------------------------------------------------------


CONVERSATION = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi, my name is Alex."},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "What is my name?"},
]


class TestRealmLabsPostCall:
    @pytest.mark.asyncio
    async def test_sends_full_conversation_plus_the_reply(self, realmlabs_guardrail):
        """MLS is stateless: the whole conversation goes on every call, with the
        model's reply appended as a trailing assistant turn."""
        handler = MagicMock(post=AsyncMock(return_value=MagicMock(**{"json.return_value": _mls_response()})))
        with patch.object(realmlabs_guardrail, "async_handler", handler):
            await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["Your name is Alex."]},
                request_data={"messages": CONVERSATION},
                input_type="response",
            )
        sent = handler.post.call_args.kwargs["json"]["messages"]
        assert sent == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi, my name is Alex."},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What is my name?"},
            {"role": "assistant", "content": "Your name is Alex."},
        ]

    @pytest.mark.asyncio
    async def test_system_prompt_is_included(self, realmlabs_guardrail):
        """The chat template injects no system prompt of its own, so ours has to
        be sent for the rendering - and therefore the offsets - to match."""
        handler = MagicMock(post=AsyncMock(return_value=MagicMock(**{"json.return_value": _mls_response()})))
        with patch.object(realmlabs_guardrail, "async_handler", handler):
            await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["reply"]},
                request_data={"messages": CONVERSATION},
                input_type="response",
            )
        roles = [m["role"] for m in handler.post.call_args.kwargs["json"]["messages"]]
        assert roles[0] == "system"

    @pytest.mark.asyncio
    async def test_multimodal_content_is_flattened_to_text(self, realmlabs_guardrail):
        handler = MagicMock(post=AsyncMock(return_value=MagicMock(**{"json.return_value": _mls_response()})))
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
                ],
            }
        ]
        with patch.object(realmlabs_guardrail, "async_handler", handler):
            await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["a picture"]},
                request_data={"messages": conversation},
                input_type="response",
            )
        assert handler.post.call_args.kwargs["json"]["messages"][0] == {
            "role": "user",
            "content": "describe this",
        }

    @pytest.mark.asyncio
    async def test_masks_pii_in_the_reply(self, realmlabs_guardrail):
        body = _mls_response(pii_spans=[{"type": "name", "text": "Alex"}])
        with _patch_mls(realmlabs_guardrail, body):
            result = await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["Your name is Alex."]},
                request_data={"messages": CONVERSATION},
                input_type="response",
            )
        assert result["texts"] == ["Your name is [name]."]

    @pytest.mark.asyncio
    async def test_hazard_is_not_enforced_on_the_response(self, realmlabs_guardrail):
        """hazard_prompt scores the user turn. On a response the focal turn is
        the assistant's, so MLS reports role_mismatch and the score must not be
        used to block the reply."""
        body = _mls_response(hazard_prob=0.9999)
        body["results"][0]["role_mismatch"] = True
        body["focal_role"] = "assistant"
        with _patch_mls(realmlabs_guardrail, body):
            result = await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["some reply"]},
                request_data={"messages": CONVERSATION},
                input_type="response",
            )
        assert result["texts"] == ["some reply"]

    @pytest.mark.asyncio
    async def test_role_mismatch_also_suppresses_hazard_on_the_request(self, realmlabs_guardrail):
        body = _mls_response(hazard_prob=0.9999)
        body["results"][0]["role_mismatch"] = True
        with _patch_mls(realmlabs_guardrail, body):
            result = await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["hello"]},
                request_data={"messages": [{"role": "user", "content": "hello"}]},
                input_type="request",
            )
        assert result["texts"] == ["hello"]

    @pytest.mark.asyncio
    async def test_no_turn_id_is_ever_sent(self, realmlabs_guardrail):
        """The endpoint is stateless - it stores nothing and takes no key."""
        handler = MagicMock(post=AsyncMock(return_value=MagicMock(**{"json.return_value": _mls_response()})))
        with patch.object(realmlabs_guardrail, "async_handler", handler):
            for input_type, request_data in (
                ("request", {"messages": CONVERSATION}),
                ("response", {"messages": CONVERSATION}),
            ):
                await realmlabs_guardrail.apply_guardrail(
                    inputs={"texts": ["x"]},
                    request_data=request_data,
                    input_type=input_type,
                )
        for call in handler.post.call_args_list:
            assert "MLS_turn_id" not in call.kwargs["json"]

    @pytest.mark.asyncio
    async def test_reply_without_conversation_still_scans(self, realmlabs_guardrail):
        """A missing/empty messages key must not silently skip the response."""
        body = _mls_response(pii_spans=[{"type": "name", "text": "Alex"}])
        with _patch_mls(realmlabs_guardrail, body):
            result = await realmlabs_guardrail.apply_guardrail(
                inputs={"texts": ["Alex was here"]},
                request_data={},
                input_type="response",
            )
        assert result["texts"] == ["[name] was here"]


# ---------------------------------------------------------------------------
# config.yaml -> constructor wiring
# ---------------------------------------------------------------------------


class TestRealmLabsInitializer:
    """Every field the config model documents must actually reach the class.

    A field declared in the config model but dropped by initialize_guardrail is
    silently ignored at runtime: the setting appears in config.yaml, the proxy
    accepts it, and nothing happens.
    """

    @staticmethod
    def _init(**litellm_param_kwargs):
        from litellm.proxy.guardrails.guardrail_hooks.realmlabs import initialize_guardrail
        from litellm.types.guardrails import LitellmParams

        params = LitellmParams(guardrail="realmlabs", mode="pre_call", **litellm_param_kwargs)
        with patch("litellm.logging_callback_manager.add_litellm_callback"):
            return initialize_guardrail(params, {"guardrail_name": "realmlabs-guard"})

    def test_all_config_fields_reach_the_guardrail(self):
        guardrail = self._init(
            api_key="cfg_key",
            api_base="https://cfg.example.com",
            probes=["hazard_prompt", "dispute"],
            hazard_threshold=0.9,
            pii=False,
            pii_mask=False,
            block_on_error=True,
            default_on=True,
        )
        assert guardrail.api_key == "cfg_key"
        assert guardrail.api_base == "https://cfg.example.com"
        assert guardrail.probes == ["hazard_prompt", "dispute"]
        assert guardrail.hazard_threshold == 0.9
        assert guardrail.pii is False
        assert guardrail.pii_mask is False
        assert guardrail.block_on_error is True
        assert guardrail.guardrail_name == "realmlabs-guard"

    def test_optional_params_are_wired(self):
        """enable_thinking/timeout live under optional_params and were dropped
        by an earlier version of the initializer."""
        guardrail = self._init(
            api_key="k",
            optional_params={"enable_thinking": True, "timeout": 42.0},
        )
        assert guardrail.enable_thinking is True
        assert guardrail.timeout == 42.0

    def test_optional_params_also_accepted_at_top_level(self):
        guardrail = self._init(api_key="k", enable_thinking=True, timeout=7.5)
        assert guardrail.enable_thinking is True
        assert guardrail.timeout == 7.5

    def test_defaults_apply_when_config_omits_them(self):
        guardrail = self._init(api_key="k")
        assert guardrail.api_base == "https://mls.realmlabs.ai"
        assert guardrail.probes == ["hazard_prompt"]
        assert guardrail.hazard_threshold == 0.703
        assert guardrail.pii is True
        assert guardrail.pii_mask is True
        assert guardrail.block_on_error is False
        assert guardrail.enable_thinking is False
        assert guardrail.timeout == 15.0

    def test_api_key_can_come_from_the_environment_instead(self):
        with patch.dict(os.environ, {"REALMLABS_API_KEY": "from_env"}):
            guardrail = self._init()
        assert guardrail.api_key == "from_env"
