from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .realmlabs import RealmLabsGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams

__all__ = ["RealmLabsGuardrail"]


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
):
    import litellm

    _realmlabs_callback: Final = RealmLabsGuardrail(
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        probes=litellm_params.probes,
        hazard_threshold=litellm_params.hazard_threshold,
        pii=litellm_params.pii,
        pii_mask=litellm_params.pii_mask,
        block_on_error=litellm_params.block_on_error,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(_realmlabs_callback)

    return _realmlabs_callback


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.REALMLABS.value: initialize_guardrail,
}


guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.REALMLABS.value: RealmLabsGuardrail,
}
