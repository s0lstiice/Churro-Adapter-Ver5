"""Opt-in scaling of individual CHURRO language LoRA projections."""

from __future__ import annotations

import re
from collections import defaultdict

from peft.tuners.lora import LoraLayer


SPEC_RE = re.compile(
    r"^(\d+):(q_proj|k_proj|v_proj|o_proj):(0(?:\.\d+)?|1(?:\.0+)?)$"
)
MODULE_RE = re.compile(
    r"language_model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)(?:\.|$)"
)


def parse_lora_component_scale_specs(
    values: list[str], layer_count: int = 36
) -> dict[tuple[int, str], float]:
    scales: dict[tuple[int, str], float] = {}
    for value in values:
        match = SPEC_RE.fullmatch(value.strip())
        if not match:
            raise ValueError(
                f"invalid LoRA component scale {value!r}; expected "
                "LAYER:PROJECTION:SCALE, e.g. 27:o_proj:0.5"
            )
        layer = int(match.group(1))
        if not 0 <= layer < layer_count:
            raise ValueError(f"LoRA layer {layer} outside 0-{layer_count - 1}")
        key = (layer, match.group(2))
        if key in scales:
            raise ValueError(f"LoRA component {layer}:{key[1]} specified more than once")
        scales[key] = float(match.group(3))
    return scales


def apply_language_lora_component_scales(
    model, scales: dict[tuple[int, str], float]
) -> dict:
    found: dict[tuple[int, str], int] = defaultdict(int)
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        match = MODULE_RE.search(name)
        if not match:
            continue
        key = (int(match.group(1)), match.group(2))
        if key not in scales:
            continue
        for adapter_name in list(module.scaling):
            module.scaling[adapter_name] *= scales[key]
        found[key] += 1
    missing = sorted(set(scales) - set(found))
    if missing:
        raise RuntimeError(f"no language LoRA modules found for components {missing}")
    return {
        "requested_scales": {
            f"{layer}:{projection}": scale
            for (layer, projection), scale in sorted(scales.items())
        },
        "scaled_module_counts": {
            f"{layer}:{projection}": found[(layer, projection)]
            for layer, projection in sorted(found)
        },
    }
