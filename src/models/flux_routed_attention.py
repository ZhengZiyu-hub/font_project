from __future__ import annotations

import torch
from torch import nn


class RoutedFluxAttnProcessor(nn.Module):
    """FLUX attention processor with routed multi-branch conditioning.

    This module does not change the FLUX transformer blocks. It replaces only
    the attention processor used by each attention layer.

    Branch inputs:
        text tokens: ``[B, L_text, D]``
        style tokens: ``[B, L_style, D]``
        glyph tokens: ``[B, L_glyph, D]``

    Runtime metadata:
        ``branch_lengths=(L_text, L_style, L_glyph)`` is passed through
        ``joint_attention_kwargs`` from the decoder wrapper.

    Learnable routing:
        each installed processor owns ``gate_logits`` with shape ``[3]``. Since
        every FLUX attention layer receives a separate processor instance, gates
        are learnable per layer. Softmax(gate_logits) weights the conditioning
        K/V tokens for text, style and glyph branches before attention.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, layer_index: int, initial_gates: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
        super().__init__()
        if len(initial_gates) != 3:
            raise ValueError("initial_gates must contain text/style/glyph weights.")
        self.layer_index = layer_index
        initial = torch.as_tensor(initial_gates, dtype=torch.float32).clamp_min(1e-6).log()
        self.gate_logits = nn.Parameter(initial)

    def branch_gates(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Return gate weights ``[3]`` for text/style/glyph.

        Softmax gives relative routing weights. Multiplying by 3 preserves the
        original conditioning magnitude when all branches start equally weighted.
        """

        return (torch.softmax(self.gate_logits.to(device=device, dtype=torch.float32), dim=0) * 3.0).to(dtype=dtype)

    def _branch_scale(
        self,
        length: int,
        branch_lengths: tuple[int, int, int] | list[int] | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if branch_lengths is None:
            return None
        if len(branch_lengths) != 3:
            raise ValueError(f"branch_lengths must contain 3 values, got {branch_lengths}.")

        text_len, style_len, glyph_len = [int(value) for value in branch_lengths]
        expected = text_len + style_len + glyph_len
        if expected > length:
            raise ValueError(f"branch_lengths sum {expected} exceeds condition length {length}.")
        if expected == 0:
            return None

        gates = self.branch_gates(dtype=dtype, device=device)
        scale = torch.ones(length, device=device, dtype=dtype)
        start = 0
        for branch_len, gate in zip((text_len, style_len, glyph_len), gates):
            if branch_len > 0:
                scale[start : start + branch_len] = gate
            start += branch_len
        return scale

    def _scale_sequence_prefix(
        self,
        tensor: torch.Tensor,
        branch_lengths: tuple[int, int, int] | list[int] | None,
    ) -> torch.Tensor:
        """Scale K/V tokens belonging to condition branches.

        ``tensor`` shape is ``[B, L, H, Dh]``. For dual-stream FLUX blocks, L is
        only condition length. For single-stream blocks, the first condition
        tokens are followed by image tokens; only that prefix is scaled.
        """

        scale = self._branch_scale(tensor.shape[1], branch_lengths, tensor.dtype, tensor.device)
        if scale is None:
            return tensor
        return tensor * scale.view(1, -1, 1, 1)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        branch_lengths: tuple[int, int, int] | list[int] | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        try:
            from diffusers.models.transformers.transformer_flux import (
                _get_qkv_projections,
                apply_rotary_emb,
                dispatch_attention_fn,
            )
        except ImportError as exc:  # pragma: no cover - depends on diffusers version.
            raise ImportError("RoutedFluxAttnProcessor requires diffusers FLUX internals.") from exc

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            # Dual-stream FLUX block: condition tokens arrive separately through
            # encoder_hidden_states. Gate branch K/V before joining them with
            # image tokens.
            encoder_key = self._scale_sequence_prefix(encoder_key, branch_lengths)
            encoder_value = self._scale_sequence_prefix(encoder_value, branch_lengths)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)
        else:
            # Single-stream FLUX block: condition and image tokens have already
            # been concatenated by the block. Gate only the condition prefix.
            key = self._scale_sequence_prefix(key, branch_lengths)
            value = self._scale_sequence_prefix(value, branch_lengths)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states

        return hidden_states


def build_routed_flux_processors(
    processor_names: list[str],
    initial_gates: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, RoutedFluxAttnProcessor]:
    """Create one routed processor per FLUX attention layer."""

    return {
        name: RoutedFluxAttnProcessor(layer_index=layer_index, initial_gates=initial_gates)
        for layer_index, name in enumerate(processor_names)
    }
