from __future__ import annotations


def apply_lora(model, *, rank: int = 64, alpha: int = 128, dropout: float = 0.05, target_modules: list[str] | None = None):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("peft is required for LoRA training") from exc
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    return get_peft_model(model, config)


def trainable_parameter_names(model) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]
