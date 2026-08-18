from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .base import GuardrailConfigModel


class RealmLabsProbeResult(TypedDict, total=False):
    """One classifier probe verdict from the RealmLabs MLS response.

    ``prob`` is the probe's score and ``threshold`` the cut-off MLS itself
    considers meaningful. LiteLLM enforces ``hazard_threshold`` from the
    guardrail config instead, so the two can differ.
    """

    probe: str
    prob: float | None
    threshold: float | None
    decision: bool | None
    expected_role: str | None
    focal_role: str | None
    role_mismatch: bool | None
    layer: int | None
    span: str | None
    pooling: str | None
    n_window_tokens: int | None


class RealmLabsPIISpan(TypedDict, total=False):
    """One detected PII span.

    ``start``/``end`` index MLS's rendered view of the whole conversation, not
    the individual message, so they are unusable for per-message masking and
    are deliberately ignored. ``text`` is located within each message instead.
    """

    type: str
    start: int | None
    end: int | None
    text: str | None
    score: float | None


class RealmLabsGuardrailResponse(TypedDict, total=False):
    """Response body of POST {api_base}/litellm/guardrail."""

    MLS_turn_id: str | None
    results: list[RealmLabsProbeResult]
    focal_role: str | None
    pii_spans: list[RealmLabsPIISpan]
    n_history_messages: int | None
    model: dict[str, Any] | None
    timings: dict[str, Any] | None


class RealmLabsGuardrailOptionalParams(BaseModel):
    """Optional parameters for the RealmLabs guardrail"""

    enable_thinking: bool | None = Field(
        default=False,
        description="Whether MLS should render the chat template in thinking mode. Defaults to False.",
    )

    timeout: float | None = Field(
        default=15.0,
        description="Timeout in seconds for the MLS request. Defaults to 15.",
    )


class RealmLabsGuardrailConfigModel(GuardrailConfigModel[RealmLabsGuardrailOptionalParams]):
    api_key: str | None = Field(
        default=None,
        description=(
            "API key for the RealmLabs MLS guardrail endpoint, sent as a bearer token. "
            "If not provided, the REALMLABS_API_KEY environment variable is used."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "Base URL of the RealmLabs MLS deployment. The /litellm/guardrail path is "
            "appended automatically. Defaults to https://mls.realmlabs.ai, and falls "
            "back to the REALMLABS_API_BASE environment variable."
        ),
    )
    probes: list[str] | str | None = Field(
        default=None,
        description=(
            'Which classifier probes to run: a list of probe names, or "all". '
            'Defaults to ["hazard_prompt"] - the only probe whose score this '
            "guardrail enforces. An unknown probe name makes MLS return 404."
        ),
    )
    hazard_threshold: float | None = Field(
        default=None,
        description=(
            "Block the request when the hazard_prompt probe scores strictly above this "
            "value. Defaults to 0.703, the threshold MLS reports for that probe. Note "
            'the probe also responds to instruction-style phrasing such as "repeat this '
            'back verbatim", so raise this if benign traffic is being blocked.'
        ),
    )
    pii: bool | None = Field(
        default=None,
        description=(
            "Whether to run MLS's PII detection head. Defaults to True. Requires "
            "VLLM_CHILD_PII_CAPTURE=1 on the MLS child process."
        ),
    )
    pii_mask: bool | None = Field(
        default=None,
        description=(
            "What to do with detected PII. True (default) rewrites each span as its "
            'type in brackets, e.g. "My name is Alex" -> "My name is [name]", and '
            "lets the request through. False blocks the request instead."
        ),
    )
    block_on_error: bool | None = Field(
        default=None,
        description=(
            "Whether to block the request when MLS is unreachable or returns an "
            "unreadable response. Defaults to False (fail open), so an MLS outage does "
            "not take the gateway down with it. Set to True to fail closed."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "RealmLabs MLS"
