"""Opt-in inter-layer visual contrast for CHURRO decoding.

This is a conservative, reference-free adaptation of image Token
attention-guided Decoding (iTaD; Xu et al., NAACL 2025,
https://aclanthology.org/2025.naacl-long.75/).  A shallow decoder state is
projected through the ordinary final norm and language head.  Its log
probabilities are subtracted from the final log probabilities, while a
plausibility cutoff prevents weak final-model candidates from being promoted.

The controller is disabled unless explicitly requested.  It changes neither
stored model weights nor the default CHURRO path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def contrast_logits(
    mature_logits: torch.Tensor,
    premature_logits: torch.Tensor,
    *,
    strength: float,
    plausibility_cutoff: float,
) -> torch.Tensor:
    """Contrast final and intermediate distributions with a mature-model gate."""

    if mature_logits.shape != premature_logits.shape:
        raise ValueError(
            f"mature shape {tuple(mature_logits.shape)} != "
            f"premature shape {tuple(premature_logits.shape)}"
        )
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    if not 0.0 < plausibility_cutoff <= 1.0:
        raise ValueError("plausibility_cutoff must be in (0, 1]")

    mature = F.log_softmax(mature_logits.float(), dim=-1)
    premature = F.log_softmax(premature_logits.float(), dim=-1)
    adjusted = mature - strength * premature
    threshold = mature.amax(dim=-1, keepdim=True) + math.log(plausibility_cutoff)
    adjusted.masked_fill_(mature < threshold, -torch.inf)
    return adjusted


def language_components(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    core = getattr(base, "model", None)
    language = getattr(core, "language_model", None)
    if language is None and hasattr(core, "model"):
        language = getattr(core.model, "language_model", None)
    if language is None:
        raise TypeError("could not locate Qwen2.5-VL language model")
    lm_head = getattr(base, "lm_head", None) or getattr(model, "lm_head", None)
    if lm_head is None:
        raise TypeError("could not locate CHURRO lm_head")
    return language, language.layers, language.norm, lm_head


@dataclass
class InterlayerVisualContrastController:
    layer: int
    strength: float
    plausibility_cutoff: float
    norm: torch.nn.Module
    lm_head: torch.nn.Module
    _hidden: torch.Tensor | None = None
    forward_calls: int = 0
    adjusted_calls: int = 0

    def capture_layer_output(self, _module, _inputs, output):
        self._hidden = output[0] if isinstance(output, tuple) else output

    def adjust_lm_head_output(self, module, _inputs, output):
        self.forward_calls += 1
        if self._hidden is None:
            return output
        if not isinstance(output, torch.Tensor) or output.ndim < 2:
            raise TypeError("CHURRO lm_head returned an unsupported output")
        output_tokens = int(output.shape[-2])
        hidden = self._hidden[..., -output_tokens:, :]
        normalized = self.norm(hidden)
        bias = getattr(module, "bias", None)
        premature = F.linear(normalized, module.weight, bias)
        adjusted = contrast_logits(
            output,
            premature,
            strength=self.strength,
            plausibility_cutoff=self.plausibility_cutoff,
        )
        self.adjusted_calls += 1
        return adjusted.to(dtype=output.dtype)

    def audit(self) -> dict:
        return {
            "schema_version": "churro_interlayer_visual_contrast.v1",
            "enabled": True,
            "intermediate_layer": self.layer,
            "strength": self.strength,
            "plausibility_cutoff": self.plausibility_cutoff,
            "forward_calls": self.forward_calls,
            "adjusted_calls": self.adjusted_calls,
            "reference_free": True,
            "weights_modified": False,
            "method_source": "Xu et al., NAACL 2025, iTaD",
        }


def install_interlayer_visual_contrast(
    model,
    *,
    layer: int,
    strength: float,
    plausibility_cutoff: float,
) -> InterlayerVisualContrastController:
    language, layers, norm, lm_head = language_components(model)
    del language
    if not 0 <= layer < len(layers) - 1:
        raise ValueError(f"intermediate layer must be in 0-{len(layers) - 2}")
    controller = InterlayerVisualContrastController(
        layer=layer,
        strength=strength,
        plausibility_cutoff=plausibility_cutoff,
        norm=norm,
        lm_head=lm_head,
    )
    controller._layer_handle = layers[layer].register_forward_hook(
        controller.capture_layer_output
    )
    controller._head_handle = lm_head.register_forward_hook(
        controller.adjust_lm_head_output
    )
    return controller
