from typing import TYPE_CHECKING, Any, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .realmlabs import RealmLabsGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams

__all__ = ["RealmLabsGuardrail"]


def _get_config_value(litellm_params: Any, optional_params: Any, attribute_name: str) -> Any | None:
    """Read a setting from optional_params, falling back to litellm_params.

    Lets a config.yaml nest tuning knobs under optional_params or set them
    directly on litellm_params, as the other guardrails do.
    """
    if optional_params is not None:
        value: Final = (
            optional_params.get(attribute_name)
            if isinstance(optional_params, dict)
            else getattr(optional_params, attribute_name, None)
        )
        if value is not None:
            return value
    return getattr(litellm_params, attribute_name, None)


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
):
    import litellm

    optional_params: Final = getattr(litellm_params, "optional_params", None)

    _realmlabs_callback: Final = RealmLabsGuardrail(
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        probes=_get_config_value(litellm_params, optional_params, "probes"),
        hazard_threshold=_get_config_value(litellm_params, optional_params, "hazard_threshold"),
        pii=_get_config_value(litellm_params, optional_params, "pii"),
        pii_mask=_get_config_value(litellm_params, optional_params, "pii_mask"),
        block_on_error=_get_config_value(litellm_params, optional_params, "block_on_error"),
        enable_thinking=_get_config_value(litellm_params, optional_params, "enable_thinking"),
        timeout=_get_config_value(litellm_params, optional_params, "timeout"),
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
