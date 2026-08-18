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

    def test_only_pre_call_is_supported(self):
        """Response-side scanning is not implemented yet."""
        assert RealmLabsGuardrail.get_supported_event_hooks() == [GuardrailEventHooks.pre_call]

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
