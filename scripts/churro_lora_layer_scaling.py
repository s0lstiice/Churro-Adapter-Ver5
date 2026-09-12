"""Opt-in per-layer scaling for CHURRO language-model LoRA adapters."""

from __future__ import annotations

import re
from collections import defaultdict

from peft.tuners.lora import LoraLayer


SPEC_RE = re.compile(r"^(\d+)(?:-(\d+))?:(0(?:\.\d+)?|1(?:\.0+)?)$")
LAYER_RE = re.compile(r"language_model\.layers\.(\d+)\.")


def parse_layer_scale_specs(values: list[str], layer_count: int = 36) -> dict[int, float]:
    scales: dict[int, float] = {}
    for value in values:
        match = SPEC_RE.fullmatch(value.strip())
        if not match:
            raise ValueError(
                f"invalid LoRA layer scale {value!r}; expected START-END:SCALE, e.g. 18-23:0.5"
            )
        start = int(match.group(1))
        stop = int(match.group(2) or start)
        scale = float(match.group(3))
        if start > stop or start < 0 or stop >= layer_count:
            raise ValueError(f"LoRA layer range {start}-{stop} outside 0-{layer_count - 1}")
        for layer in range(start, stop + 1):
            if layer in scales:
                raise ValueError(f"LoRA layer {layer} is specified more than once")
            scales[layer] = scale
    return scales


def apply_language_lora_layer_scales(model, scales: dict[int, float]) -> dict:
    found: dict[int, int] = defaultdict(int)
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        match = LAYER_RE.search(name)
        if not match:
            continue
        layer = int(match.group(1))
        if layer not in scales:
            continue
        for adapter_name in list(module.scaling):
            module.scaling[adapter_name] *= scales[layer]
        found[layer] += 1
    missing = sorted(set(scales) - set(found))
    if missing:
        raise RuntimeError(f"no language LoRA modules found for requested layers {missing}")
    return {
        "requested_scales": {str(layer): scale for layer, scale in sorted(scales.items())},
        "scaled_module_counts": {str(layer): found[layer] for layer in sorted(found)},
    }
