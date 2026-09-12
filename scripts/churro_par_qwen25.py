#!/usr/bin/env python3
"""Qwen2.5-VL port of PAR for visually faithful CHURRO decoding.

PAR (Positional Perturbation and Attention Recycling) is a training-free OCR
decoding intervention described by Yao et al., ACL 2026:
https://aclanthology.org/2026.acl-long.1065/

The authors' reference implementation targets Qwen3-VL.  This module is an
independent, minimal port for the Qwen2.5-VL implementation used by CHURRO.  It
patches selected attention modules in place so loaded PEFT/LoRA projection
modules remain exactly where they are.  Prefill is delegated to the original
attention implementation; the intervention runs only for one-token cached
decoding.
"""

from __future__ import annotations

import math
import types
from functools import lru_cache
from dataclasses import dataclass, field, replace
from typing import Iterable

import torch
import torch.nn.functional as F

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)


@dataclass(frozen=True)
class PARConfig:
    """Inference-time PAR settings.

    Defaults follow the public Qwen OCR configuration where architecture
    permits.  ``seed`` makes positional perturbation repeatable per page.
    """

    pp_enabled: bool = True
    far_enabled: bool = True
    spin_enabled: bool = False
    target_layers: tuple[int, ...] = (0, 1, 2, 3)
    seed: int = 1729

    pp_alpha: float = 0.05
    pp_fixed_freq: float = 1.0
    pp_peak_pos: float = 300.0
    pp_rise_tau: float = 100.0
    pp_decay_tau: float = 500.0
    pp_floor_ratio: float = 0.05

    far_outlier_std_threshold: float = 3.0
    far_base_penalty_strength: float = 0.1
    far_max_penalty_add: float = 0.7
    far_saturation_len: float = 300.0
    far_salience_threshold_sigma: float = 1.5
    far_gaussian_sigma: float = 20.0
    far_num_gaussian_sigma: float = 4.0
    far_refill_factor: float = 1.5

    # SPIN-style dynamic visual-head suppression. At each cached decoding
    # step, preserve the heads with the most image attention and attenuate the
    # remainder. Unlike the earlier static head ablations, membership is
    # recomputed for every token and layer.
    spin_keep_fraction: float = 0.90
    spin_suppression_scale: float = 0.50

    def __post_init__(self) -> None:
        if not self.target_layers:
            raise ValueError("PAR target_layers cannot be empty")
        if self.pp_alpha < 0:
            raise ValueError("pp_alpha must be non-negative")
        if self.pp_rise_tau <= 0 or self.pp_decay_tau <= 0:
            raise ValueError("PP time constants must be positive")
        if self.far_saturation_len <= 0 or self.far_gaussian_sigma <= 0:
            raise ValueError("FAR saturation length and Gaussian sigma must be positive")
        if not 0 < self.spin_keep_fraction <= 1:
            raise ValueError("spin_keep_fraction must be in (0,1]")
        if not 0 <= self.spin_suppression_scale <= 1:
            raise ValueError("spin_suppression_scale must be in [0,1]")


@dataclass
class _LayerState:
    layer_idx: int
    config: PARConfig
    original_forward: object
    image_mask: torch.Tensor | None = None
    prompt_length: int = 0
    generator: torch.Generator | None = None
    generator_device: torch.device | None = None
    decode_steps: int = 0
    far_steps: torch.Tensor | None = None
    recycled_mass_sum: torch.Tensor | None = None
    capture_visual_attention: bool = False
    visual_attention_sum: torch.Tensor | None = None
    visual_attention_steps: int = 0
    trace_visual_grounding: bool = False
    visual_grounding_trace: list[torch.Tensor] = field(default_factory=list)
    audit_enabled: bool = True
    far_audit_excluded_decode_steps: int = 0
    spin_steps: int = 0
    spin_suppressed_heads_sum: int = 0

    def begin_page(self, image_mask: torch.Tensor) -> None:
        self.image_mask = image_mask.detach()
        self.prompt_length = int(image_mask.shape[-1])
        self.generator = None
        self.generator_device = None
        self.decode_steps = 0
        self.far_steps = None
        self.recycled_mass_sum = None
        self.capture_visual_attention = False
        self.visual_attention_sum = None
        self.visual_attention_steps = 0
        self.trace_visual_grounding = False
        self.visual_grounding_trace = []
        self.audit_enabled = True
        self.far_audit_excluded_decode_steps = 0
        self.spin_steps = 0
        self.spin_suppressed_heads_sum = 0

    def random_phase(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.generator is None or self.generator_device != device:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(self.config.seed + 1009 * self.layer_idx)
            self.generator_device = device
        return torch.randn(shape, device=device, dtype=dtype, generator=self.generator)


@dataclass(frozen=True)
class PARAuditCheckpoint:
    """Rollback-safe snapshot of non-causal PAR accounting state."""

    layers: tuple[
        tuple[int, torch.Tensor | None, torch.Tensor | None, int, int, int], ...
    ]


def _resolve_qwen25_core(model):
    candidates = []
    if hasattr(model, "get_base_model"):
        try:
            candidates.append(model.get_base_model())
        except Exception:
            pass
    candidates.extend([model, getattr(model, "model", None), getattr(model, "base_model", None)])
    seen = set()
    while candidates:
        candidate = candidates.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        nested_model = getattr(candidate, "model", None)
        language_model = getattr(nested_model, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if layers is not None:
            return candidate, layers
        for name in ("model", "base_model"):
            nested = getattr(candidate, name, None)
            if nested is not None and id(nested) not in seen:
                candidates.append(nested)
    raise TypeError("Could not locate Qwen2.5-VL language_model.layers")


def _pp_position_embeddings(
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor | None,
    state: _LayerState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply repeatable positional perturbation to one decode position."""

    cfg = state.config
    if cache_position is None:
        position = torch.tensor([state.prompt_length + state.decode_steps], device=cos.device)
    else:
        position = cache_position.to(device=cos.device)
    # Qwen2.5-VL cos/sin are [3, batch, query, head_dim].  A
    # [1, batch, query, 1] perturbation broadcasts across mRoPE axes and head
    # dimensions while retaining the original temporal/height/width encoding.
    batch = int(cos.shape[1])
    query = int(cos.shape[-2])
    pos = position[-query:].to(dtype=torch.float32).view(1, 1, query, 1).expand(1, batch, query, 1)
    phase = state.random_phase((1, batch, query, 1), cos.device, torch.float32)
    base_pattern = torch.sin(pos * cfg.pp_fixed_freq + phase)
    rise = torch.tanh(pos / cfg.pp_rise_tau)
    after_peak = torch.clamp(pos - cfg.pp_peak_pos, min=0.0)
    decay = cfg.pp_floor_ratio + (1.0 - cfg.pp_floor_ratio) * torch.exp(
        -after_peak / cfg.pp_decay_tau
    )
    noise = (cfg.pp_alpha * rise * decay * base_pattern).to(dtype=cos.dtype)
    return cos + sin * noise, sin - cos * noise


@lru_cache(maxsize=16)
def _gaussian_kernel(config: PARConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma = config.far_gaussian_sigma
    size = int(config.far_num_gaussian_sigma * sigma + 1)
    if size % 2 == 0:
        size += 1
    grid = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    kernel = torch.exp(-0.5 * (grid / sigma).square())
    kernel = kernel / kernel.sum().clamp_min(1e-6)
    return kernel.to(dtype=dtype).view(1, 1, -1)


def _recycle_attention(
    logits: torch.Tensor,
    image_mask: torch.Tensor,
    prompt_length: int,
    config: PARConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move abnormal generated-text attention mass to salient image tokens."""

    key_length = int(logits.shape[-1])
    effective = min(int(image_mask.shape[-1]), key_length)
    if effective <= 0 or key_length <= prompt_length + 1:
        zero = logits.new_zeros((), dtype=torch.float32)
        return logits, zero, zero

    valid_image = image_mask[:, :effective].to(device=logits.device, dtype=torch.bool)
    image_mask_4d = valid_image[:, None, None, :]
    image_float = image_mask_4d.to(dtype=logits.dtype)
    image_logits = logits[..., :effective]
    image_count = image_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
    image_mean = (image_logits * image_float).sum(dim=-1, keepdim=True) / image_count
    image_second = (image_logits.square() * image_float).sum(dim=-1, keepdim=True) / image_count
    image_std = (image_second - image_mean.square()).clamp_min(1e-6).sqrt()

    # Only previously generated response tokens are eligible for suppression.
    # System/user prompt tokens and every image token remain protected.
    text_mask = torch.zeros(
        (logits.shape[0], 1, 1, key_length), dtype=torch.bool, device=logits.device
    )
    text_mask[..., prompt_length:key_length] = True
    text_float = text_mask.to(dtype=logits.dtype)
    text_count = text_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
    text_mean = (logits * text_float).sum(dim=-1, keepdim=True) / text_count
    text_second = (logits.square() * text_float).sum(dim=-1, keepdim=True) / text_count
    text_std = (text_second - text_mean.square()).clamp_min(1e-6).sqrt()
    threshold = text_mean + config.far_outlier_std_threshold * text_std
    excess = F.relu(logits - threshold) * text_float

    generated_length = max(0, key_length - prompt_length)
    progress = min(1.0, generated_length / config.far_saturation_len)
    penalty_strength = config.far_base_penalty_strength + config.far_max_penalty_add * math.sqrt(progress)
    penalty = excess * penalty_strength
    total_penalty = penalty.sum(dim=-1, keepdim=True)

    salience_threshold = image_mean + config.far_salience_threshold_sigma * image_std
    raw_salience = F.relu(image_logits - salience_threshold) * image_float
    flat = raw_salience.reshape(-1, 1, effective)
    kernel = _gaussian_kernel(config, logits.device, logits.dtype)
    diffused = F.conv1d(flat, kernel, padding=kernel.shape[-1] // 2).reshape_as(raw_salience)
    diffused = diffused * image_float
    weight_sum = diffused.sum(dim=-1, keepdim=True)
    # Gate independently per attention head.  A head without a salient image
    # destination must retain its original text logits even if another head in
    # the same layer can recycle attention.  Keeping this tensor-side also
    # avoids a GPU-to-CPU synchronization at every generated token.
    usable = (weight_sum > 1e-6) & (total_penalty > 0)
    usable_float = usable.to(dtype=logits.dtype)
    applied_penalty = penalty * usable_float
    applied_total_penalty = total_penalty * usable_float
    adjusted = logits - applied_penalty
    normalized = diffused / weight_sum.clamp_min(1e-6)
    refill = applied_total_penalty * normalized * config.far_refill_factor
    adjusted_image = adjusted[..., :effective] + refill
    adjusted = torch.cat((adjusted_image, adjusted[..., effective:]), dim=-1)
    recycled_mass = applied_total_penalty.detach().float().mean()
    recycled_step = usable.detach().any().to(dtype=torch.float32)
    return adjusted, recycled_mass, recycled_step


def _raw_image_attention_mass(
    logits: torch.Tensor,
    image_mask: torch.Tensor,
) -> torch.Tensor:
    """Measure pre-intervention image attention for each batch branch.

    The returned vector is averaged over heads and query positions but not
    over batch.  It is captured only during a bounded rollback replay, so the
    additional softmax is not paid during ordinary page decoding.  Measuring
    before FAR is important: FAR deliberately moves mass onto image tokens and
    therefore cannot itself be used as independent grounding evidence.
    """

    effective = min(int(image_mask.shape[-1]), int(logits.shape[-1]))
    if effective <= 0:
        return logits.new_zeros((logits.shape[0],), dtype=torch.float32)
    valid_image = image_mask[:, :effective].to(device=logits.device, dtype=torch.bool)
    weights = F.softmax(logits, dim=-1, dtype=torch.float32)
    image_weights = weights[..., :effective] * valid_image[:, None, None, :]
    return image_weights.sum(dim=-1).mean(dim=(1, 2))


def _normalized_visual_grounding(
    logits: torch.Tensor,
    image_mask: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """Measure image attention per token relative to generated-history attention.

    Raw image-attention mass naturally falls as a response grows because more
    generated keys compete in the denominator.  Dividing mean per-image-token
    attention by mean per-generated-token attention makes the trace useful for
    detecting a real shift from reading pixels to following self-generated
    prose near the end of a page.
    """

    effective = min(int(image_mask.shape[-1]), int(logits.shape[-1]))
    key_length = int(logits.shape[-1])
    if effective <= 0 or key_length <= prompt_length:
        return logits.new_zeros((logits.shape[0],), dtype=torch.float32)
    weights = F.softmax(logits, dim=-1, dtype=torch.float32)
    valid_image = image_mask[:, :effective].to(device=logits.device, dtype=torch.bool)
    image_count = valid_image.sum(dim=-1).clamp_min(1).to(dtype=torch.float32)
    image_mass = (
        weights[..., :effective] * valid_image[:, None, None, :]
    ).sum(dim=-1)
    generated = weights[..., prompt_length:key_length]
    generated_count = max(1, key_length - prompt_length)
    generated_mass = generated.sum(dim=-1)
    image_mean = image_mass / image_count[:, None, None]
    generated_mean = generated_mass / float(generated_count)
    ratio = image_mean / generated_mean.clamp_min(1e-9)
    return ratio.mean(dim=(1, 2))


def _suppress_low_visual_heads(
    head_outputs: torch.Tensor,
    attention_scores: torch.Tensor,
    image_mask: torch.Tensor,
    *,
    keep_fraction: float,
    suppression_scale: float,
    image_present: bool | None = None,
) -> tuple[torch.Tensor, int]:
    """Attenuate the dynamically least visual heads for this decode token.

    ``head_outputs`` is ``[batch, heads, query, head_dim]`` and
    ``attention_scores`` is the pre-softmax QK score tensor with shape
    ``[batch, heads, query, keys]``. Selection follows SPIN's accumulated raw
    attention score over actual image tokens, never a fixed head identity. The
    operation is deliberately fail-open when no image token is present.
    """

    effective = min(int(image_mask.shape[-1]), int(attention_scores.shape[-1]))
    if effective <= 0:
        return head_outputs, 0
    # The controller validates this once per page. Avoid a GPU-to-CPU
    # synchronization in every layer at every decoded token.
    if image_present is None:
        image_present = bool(image_mask[:, :effective].any())
    if not image_present:
        return head_outputs, 0
    head_count = int(attention_scores.shape[1])
    keep = max(1, min(head_count, math.ceil(head_count * keep_fraction)))
    if keep >= head_count or suppression_scale >= 1.0:
        return head_outputs, 0
    valid_image = image_mask[:, :effective].to(
        device=attention_scores.device, dtype=torch.bool
    )
    # The causal/additive attention mask can leave ``-inf`` scores outside the
    # image span. Multiplication by a false boolean is unsafe there because
    # ``-inf * 0`` becomes NaN and corrupts the head ranking. Select image
    # positions with ``masked_fill`` instead, which keeps the raw-QK SPIN
    # ranking while making every non-image position an exact zero.
    visual_scores = attention_scores[..., :effective].masked_fill(
        ~valid_image[:, None, None, :], 0.0
    )
    visual_mass = visual_scores.sum(dim=-1).mean(dim=-1)
    top = visual_mass.topk(keep, dim=1, largest=True, sorted=False).indices
    preserved = torch.zeros_like(visual_mass, dtype=torch.bool)
    preserved.scatter_(1, top, True)
    multiplier = torch.where(
        preserved,
        torch.ones_like(visual_mass),
        torch.full_like(visual_mass, suppression_scale),
    )
    adjusted = head_outputs * multiplier[:, :, None, None].to(dtype=head_outputs.dtype)
    return adjusted, head_count - keep


def _par_forward(
    self: Qwen2_5_VLAttention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
):
    state: _LayerState = self._churro_par_state
    if (
        hidden_states.shape[1] != 1
        or state.image_mask is None
        or not (
            state.config.pp_enabled
            or state.config.far_enabled
            or state.config.spin_enabled
            or state.capture_visual_attention
            or state.trace_visual_grounding
        )
    ):
        return state.original_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    if position_embeddings is None:
        raise ValueError("PAR requires Qwen2.5-VL position_embeddings")
    bsz, q_len, _ = hidden_states.shape
    query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    if state.config.pp_enabled:
        cos, sin = _pp_position_embeddings(cos, sin, cache_position, state)
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        self.config.rope_parameters["mrope_section"],
    )
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    logits = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
        logits = logits + attention_mask[..., : key_states.shape[-2]]

    if state.capture_visual_attention:
        visual_mass = _raw_image_attention_mass(logits, state.image_mask)
        if state.visual_attention_sum is None:
            state.visual_attention_sum = visual_mass
        else:
            state.visual_attention_sum = state.visual_attention_sum + visual_mass
        state.visual_attention_steps += 1

    if state.trace_visual_grounding:
        state.visual_grounding_trace.append(
            _normalized_visual_grounding(logits, state.image_mask, state.prompt_length).detach()
        )

    if state.config.far_enabled:
        logits, recycled_mass, recycled = _recycle_attention(
            logits, state.image_mask, state.prompt_length, state.config
        )
        if state.audit_enabled:
            if state.far_steps is None:
                state.far_steps = recycled
                state.recycled_mass_sum = recycled_mass
            else:
                state.far_steps = state.far_steps + recycled
                state.recycled_mass_sum = state.recycled_mass_sum + recycled_mass
    weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(dtype=query_states.dtype)
    output = torch.matmul(weights, value_states)
    if state.config.spin_enabled:
        output, suppressed_heads = _suppress_low_visual_heads(
            output,
            logits,
            state.image_mask,
            keep_fraction=state.config.spin_keep_fraction,
            suppression_scale=state.config.spin_suppression_scale,
            image_present=True,
        )
        if state.audit_enabled:
            state.spin_steps += 1
            state.spin_suppressed_heads_sum += suppressed_heads
    output = output.transpose(1, 2).contiguous()
    output = output.reshape(bsz, q_len, -1)
    output = self.o_proj(output)
    if state.audit_enabled:
        state.decode_steps += 1
    return output, weights


@dataclass
class PARController:
    """Owns the in-place patches and resets them for each page."""

    model: object
    config: PARConfig
    image_token_ids: tuple[int, ...]
    states: list[_LayerState] = field(default_factory=list)

    def configure(
        self,
        *,
        pp_enabled: bool,
        far_enabled: bool,
        spin_enabled: bool | None = None,
    ) -> None:
        """Switch intervention components without reloading model weights."""

        self.config = replace(
            self.config,
            pp_enabled=pp_enabled,
            far_enabled=far_enabled,
            spin_enabled=(self.config.spin_enabled if spin_enabled is None else spin_enabled),
        )
        for state in self.states:
            state.config = self.config

    def begin_page(self, input_ids: torch.Tensor) -> None:
        image_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.image_token_ids:
            image_mask |= input_ids.eq(token_id)
        if not bool(image_mask.any()):
            raise ValueError("PAR could not find image tokens in the rendered CHURRO prompt")
        for state in self.states:
            state.begin_page(image_mask)

    def start_visual_capture(self) -> None:
        """Capture branch-local, pre-FAR image attention until stopped."""

        for state in self.states:
            state.capture_visual_attention = True
            state.visual_attention_sum = None
            state.visual_attention_steps = 0

    def stop_visual_capture(self) -> torch.Tensor:
        """Stop capture and return one mean image-attention value per branch."""

        captured = []
        for state in self.states:
            state.capture_visual_attention = False
            if state.visual_attention_sum is not None and state.visual_attention_steps:
                captured.append(state.visual_attention_sum / state.visual_attention_steps)
        if not captured:
            raise RuntimeError("no visual-attention samples were captured")
        return torch.stack(captured, dim=0).mean(dim=0)

    def start_grounding_trace(self) -> None:
        """Record normalized visual grounding for every cached decode step."""

        for state in self.states:
            state.trace_visual_grounding = True
            state.visual_grounding_trace = []

    def stop_grounding_trace(self) -> torch.Tensor:
        """Return ``[decode_steps, batch]`` grounding averaged across layers."""

        layer_traces = []
        for state in self.states:
            state.trace_visual_grounding = False
            if state.visual_grounding_trace:
                layer_traces.append(torch.stack(state.visual_grounding_trace, dim=0))
        if not layer_traces:
            raise RuntimeError("no visual-grounding trace was captured")
        minimum_steps = min(int(trace.shape[0]) for trace in layer_traces)
        if minimum_steps <= 0:
            raise RuntimeError("visual-grounding trace is empty")
        aligned = [trace[:minimum_steps] for trace in layer_traces]
        return torch.stack(aligned, dim=0).mean(dim=0)

    def audit_checkpoint(self) -> PARAuditCheckpoint:
        """Snapshot counters at a KV rollback boundary."""

        return PARAuditCheckpoint(
            layers=tuple(
                (
                    state.decode_steps,
                    state.far_steps.detach().clone() if state.far_steps is not None else None,
                    (
                        state.recycled_mass_sum.detach().clone()
                        if state.recycled_mass_sum is not None
                        else None
                    ),
                    state.far_audit_excluded_decode_steps,
                    state.spin_steps,
                    state.spin_suppressed_heads_sum,
                )
                for state in self.states
            )
        )

    def restore_audit_checkpoint(self, checkpoint: PARAuditCheckpoint) -> None:
        if len(checkpoint.layers) != len(self.states):
            raise ValueError("PAR audit checkpoint does not match installed layers")
        for state, values in zip(self.states, checkpoint.layers):
            (
                decode_steps,
                far_steps,
                recycled_mass_sum,
                excluded,
                spin_steps,
                spin_suppressed_heads_sum,
            ) = values
            state.decode_steps = int(decode_steps)
            state.far_steps = far_steps.detach().clone() if far_steps is not None else None
            state.recycled_mass_sum = (
                recycled_mass_sum.detach().clone() if recycled_mass_sum is not None else None
            )
            state.far_audit_excluded_decode_steps = int(excluded)
            state.spin_steps = int(spin_steps)
            state.spin_suppressed_heads_sum = int(spin_suppressed_heads_sum)

    def set_audit_enabled(self, enabled: bool) -> None:
        for state in self.states:
            state.audit_enabled = bool(enabled)

    def account_selected_replay(self, decode_steps: int) -> None:
        """Advance causal step counts after a branch-only replay.

        Branch comparison runs with accounting suspended because the FAR
        tensors are batch aggregates and cannot be assigned to the selected
        branch afterward.  This keeps ``decode_steps`` exact and explicitly
        records the small region excluded from FAR-only diagnostic averages.
        """

        if decode_steps < 0:
            raise ValueError("decode_steps must be non-negative")
        for state in self.states:
            state.decode_steps += int(decode_steps)
            state.far_audit_excluded_decode_steps += int(decode_steps)

    def audit(self) -> dict:
        far_steps = {
            str(state.layer_idx): (
                int(state.far_steps.detach().cpu().item())
                if state.far_steps is not None
                else 0
            )
            for state in self.states
        }
        recycled_mass_sums = {
            str(state.layer_idx): (
                float(state.recycled_mass_sum.detach().cpu().item())
                if state.recycled_mass_sum is not None
                else 0.0
            )
            for state in self.states
        }
        return {
            "pp_enabled": self.config.pp_enabled,
            "far_enabled": self.config.far_enabled,
            "spin_enabled": self.config.spin_enabled,
            "spin_keep_fraction": self.config.spin_keep_fraction,
            "spin_suppression_scale": self.config.spin_suppression_scale,
            "target_layers": list(self.config.target_layers),
            "seed": self.config.seed,
            "decode_steps_by_layer": {str(s.layer_idx): s.decode_steps for s in self.states},
            "far_steps_by_layer": far_steps,
            "mean_recycled_mass_by_layer": {
                str(state.layer_idx): (
                    recycled_mass_sums[str(state.layer_idx)] / far_steps[str(state.layer_idx)]
                    if far_steps[str(state.layer_idx)]
                    else 0.0
                )
                for state in self.states
            },
            "far_audit_excluded_decode_steps_by_layer": {
                str(state.layer_idx): state.far_audit_excluded_decode_steps
                for state in self.states
            },
            "spin_steps_by_layer": {
                str(state.layer_idx): state.spin_steps for state in self.states
            },
            "mean_suppressed_heads_by_layer": {
                str(state.layer_idx): (
                    state.spin_suppressed_heads_sum / state.spin_steps
                    if state.spin_steps else 0.0
                )
                for state in self.states
            },
        }


def install_par(model, config: PARConfig | None = None) -> PARController:
    """Patch selected CHURRO/Qwen2.5-VL self-attention modules in place."""

    config = config or PARConfig()
    core, layers = _resolve_qwen25_core(model)
    layer_count = len(layers)
    invalid = [index for index in config.target_layers if index < 0 or index >= layer_count]
    if invalid:
        raise IndexError(f"PAR layers {invalid} outside Qwen2.5-VL layer count {layer_count}")
    model_config = core.config
    token_ids = tuple(
        int(value)
        for value in (
            getattr(model_config, "image_token_id", None),
            getattr(model_config, "video_token_id", None),
        )
        if value is not None
    )
    controller = PARController(model=model, config=config, image_token_ids=token_ids)
    for index in config.target_layers:
        attention = layers[index].self_attn
        if not isinstance(attention, Qwen2_5_VLAttention):
            raise TypeError(
                f"layer {index} self_attn is {type(attention).__name__}, expected Qwen2_5_VLAttention"
            )
        if hasattr(attention, "_churro_par_state"):
            raise RuntimeError(f"PAR is already installed on layer {index}")
        state = _LayerState(index, config, attention.forward)
        attention._churro_par_state = state
        attention.forward = types.MethodType(_par_forward, attention)
        controller.states.append(state)
    return controller


def parse_layers(value: str | Iterable[int]) -> tuple[int, ...]:
    if not isinstance(value, str):
        return tuple(int(item) for item in value)
    layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not layers:
        raise ValueError("at least one PAR layer is required")
    return layers
