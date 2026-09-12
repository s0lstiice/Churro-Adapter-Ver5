"""Opt-in scaling of selected CHURRO language attention-head outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch


SPEC_RE = re.compile(r"^(\d+):(\d+):(0(?:\.\d+)?|1(?:\.0+)?)$")


def parse_attention_head_scale_specs(values: list[str]) -> dict[tuple[int, int], float]:
    scales: dict[tuple[int, int], float] = {}
    for value in values:
        match = SPEC_RE.fullmatch(value.strip())
        if not match:
            raise ValueError(
                f"invalid attention-head scale {value!r}; expected LAYER:HEAD:SCALE, "
                "for example 33:4:0.0"
            )
        key = (int(match.group(1)), int(match.group(2)))
        if key in scales:
            raise ValueError(f"attention head {key[0]}:{key[1]} is specified more than once")
        scales[key] = float(match.group(3))
    return scales


def scale_head_slices(
    hidden: torch.Tensor, scales: dict[int, float], head_dim: int
) -> torch.Tensor:
    if hidden.shape[-1] % head_dim:
        raise ValueError(
            f"hidden width {hidden.shape[-1]} is not divisible by head_dim {head_dim}"
        )
    head_count = hidden.shape[-1] // head_dim
    invalid = sorted(head for head in scales if not 0 <= head < head_count)
    if invalid:
        raise IndexError(f"heads {invalid} outside 0-{head_count - 1}")
    output = hidden.clone()
    for head, scale in scales.items():
        output[..., head * head_dim : (head + 1) * head_dim] *= scale
    return output


def make_head_scale_pre_hook(scales: dict[int, float], head_dim: int):
    def hook(_module, inputs):
        if not inputs:
            raise RuntimeError("attention output projection received no positional input")
        return (scale_head_slices(inputs[0], scales, head_dim), *inputs[1:])

    return hook


def language_layers(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    core = getattr(base, "model", None)
    language = getattr(core, "language_model", None)
    if language is None and hasattr(core, "model"):
        language = getattr(core.model, "language_model", None)
    if language is None:
        raise TypeError("could not locate Qwen2.5-VL language_model")
    return language.layers


@dataclass
class AttentionHeadScaleController:
    handles: list
    audit: dict

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def install_attention_head_scales(
    model, scales: dict[tuple[int, int], float]
) -> AttentionHeadScaleController:
    layers = language_layers(model)
    by_layer: dict[int, dict[int, float]] = {}
    for (layer, head), scale in scales.items():
        by_layer.setdefault(layer, {})[head] = scale
    invalid_layers = sorted(layer for layer in by_layer if not 0 <= layer < len(layers))
    if invalid_layers:
        raise IndexError(f"language layers {invalid_layers} outside 0-{len(layers) - 1}")
    handles = []
    audit_layers = {}
    for layer_index, head_scales in sorted(by_layer.items()):
        attention = layers[layer_index].self_attn
        config = attention.config
        head_count = int(config.num_attention_heads)
        head_dim = int(getattr(config, "head_dim", config.hidden_size // head_count))
        invalid_heads = sorted(head for head in head_scales if not 0 <= head < head_count)
        if invalid_heads:
            raise IndexError(
                f"layer {layer_index} heads {invalid_heads} outside 0-{head_count - 1}"
            )
        handles.append(
            attention.o_proj.register_forward_pre_hook(
                make_head_scale_pre_hook(dict(head_scales), head_dim)
            )
        )
        audit_layers[str(layer_index)] = {
            "head_dim": head_dim,
            "head_count": head_count,
            "scales": {str(head): scale for head, scale in sorted(head_scales.items())},
        }
    return AttentionHeadScaleController(
        handles=handles,
        audit={"layers": audit_layers, "implementation": "pre-o_proj head-output scaling"},
    )
