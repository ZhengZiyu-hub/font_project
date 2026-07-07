from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import torch
from torch import nn


class GlyphEncoder(nn.Module):
    """Text-only glyph prior encoder for branch A.

    Input:
        text_prompt: ``str`` or ``list[str]``. No external content image is
            required.

    Output:
        glyph_tokens: ``[B, num_tokens, hidden_dim]``. These tokens are already
            in the condition dimension used by FLUX cross-attention.

    Design:
        Each character is represented by Unicode codepoint bytes, Unicode block,
        coarse character category, optional radical id and position embedding.
        A small Transformer contextualizes the character sequence, then learnable
        query tokens cross-attend to produce a fixed number of glyph tokens.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_tokens: int = 32,
        max_length: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        radical_vocab_size: int = 256,
        radical_map_path: str | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")

        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.max_length = max_length
        self.radical_vocab_size = radical_vocab_size
        self.radical_map = self._load_radical_map(radical_map_path)

        # Four byte embeddings cover every Unicode codepoint up to U+10FFFF.
        self.byte_embedding = nn.Embedding(256, hidden_dim)
        self.block_embedding = nn.Embedding(16, hidden_dim)
        self.category_embedding = nn.Embedding(32, hidden_dim)
        self.radical_embedding = nn.Embedding(radical_vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.query_tokens = nn.Parameter(torch.randn(num_tokens, hidden_dim) * 0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.char_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def _load_radical_map(self, radical_map_path: str | None) -> dict[str, int]:
        """Load optional character-to-radical mapping.

        Expected JSON format:
            ``{"汉": 85, "字": 39}``

        If no map is provided, the encoder still works with Unicode/block/category
        features and a deterministic radical bucket fallback.
        """

        if not radical_map_path:
            return {}
        path = Path(radical_map_path)
        if not path.exists():
            raise FileNotFoundError(f"radical_map_path does not exist: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(key): int(value) % self.radical_vocab_size for key, value in data.items()}

    def _unicode_block_id(self, codepoint: int) -> int:
        """Coarse Unicode block id for glyph structure hints."""

        if 0x4E00 <= codepoint <= 0x9FFF:
            return 1  # CJK Unified Ideographs
        if 0x3400 <= codepoint <= 0x4DBF:
            return 2  # CJK Extension A
        if 0x3040 <= codepoint <= 0x30FF:
            return 3  # Japanese kana
        if 0xAC00 <= codepoint <= 0xD7AF:
            return 4  # Hangul syllables
        if 0x0030 <= codepoint <= 0x007A:
            return 5  # Latin letters / digits
        if 0x2E80 <= codepoint <= 0x2FDF:
            return 6  # CJK radicals
        return 0

    def _category_id(self, char: str) -> int:
        """Map Unicode general category to a compact embedding id."""

        category = unicodedata.category(char)
        major = category[0] if category else "?"
        return {
            "L": 1,  # Letter
            "M": 2,  # Mark
            "N": 3,  # Number
            "P": 4,  # Punctuation
            "S": 5,  # Symbol
            "Z": 6,  # Separator
        }.get(major, 0)

    def _radical_id(self, char: str) -> int:
        """Return radical id from an optional map, with deterministic fallback."""

        if char in self.radical_map:
            return self.radical_map[char]
        codepoint = ord(char)
        if 0x2E80 <= codepoint <= 0x2FDF:
            return (codepoint - 0x2E80 + 1) % self.radical_vocab_size
        if 0x4E00 <= codepoint <= 0x9FFF:
            return (codepoint % (self.radical_vocab_size - 1)) + 1
        return 0

    def _normalize_prompt(self, text_prompt: str | list[str]) -> list[str]:
        if isinstance(text_prompt, str):
            return [text_prompt]
        if not text_prompt:
            raise ValueError("text_prompt list must not be empty")
        return text_prompt

    def _tokenize(self, text_prompt: str | list[str], device: torch.device) -> tuple[torch.Tensor, ...]:
        prompts = self._normalize_prompt(text_prompt)
        batch_size = len(prompts)
        bytes_tensor = torch.zeros(batch_size, self.max_length, 4, dtype=torch.long, device=device)
        block_ids = torch.zeros(batch_size, self.max_length, dtype=torch.long, device=device)
        category_ids = torch.zeros(batch_size, self.max_length, dtype=torch.long, device=device)
        radical_ids = torch.zeros(batch_size, self.max_length, dtype=torch.long, device=device)
        valid_mask = torch.zeros(batch_size, self.max_length, dtype=torch.bool, device=device)

        for batch_idx, prompt in enumerate(prompts):
            for char_idx, char in enumerate(prompt[: self.max_length]):
                codepoint = ord(char)
                bytes_tensor[batch_idx, char_idx, 0] = codepoint & 0xFF
                bytes_tensor[batch_idx, char_idx, 1] = (codepoint >> 8) & 0xFF
                bytes_tensor[batch_idx, char_idx, 2] = (codepoint >> 16) & 0xFF
                bytes_tensor[batch_idx, char_idx, 3] = (codepoint >> 24) & 0xFF
                block_ids[batch_idx, char_idx] = self._unicode_block_id(codepoint)
                category_ids[batch_idx, char_idx] = self._category_id(char)
                radical_ids[batch_idx, char_idx] = self._radical_id(char)
                valid_mask[batch_idx, char_idx] = True

        return bytes_tensor, block_ids, category_ids, radical_ids, valid_mask

    def forward(self, text_prompt: str | list[str]) -> torch.Tensor:
        device = self.query_tokens.device
        bytes_tensor, block_ids, category_ids, radical_ids, valid_mask = self._tokenize(text_prompt, device=device)
        positions = torch.arange(self.max_length, device=device).unsqueeze(0)

        # Character tokens: [B, L, D]. Codepoint byte embeddings are summed so
        # rare Unicode characters remain representable without a huge vocabulary.
        char_tokens = self.byte_embedding(bytes_tensor).sum(dim=2)
        char_tokens = char_tokens + self.block_embedding(block_ids)
        char_tokens = char_tokens + self.category_embedding(category_ids)
        char_tokens = char_tokens + self.radical_embedding(radical_ids)
        char_tokens = char_tokens + self.position_embedding(positions)
        char_tokens = self.char_norm(char_tokens)

        # True means padding for nn.TransformerEncoder. Empty prompts are allowed
        # by keeping the first token attendable as an all-zero structural token.
        padding_mask = ~valid_mask
        empty_rows = padding_mask.all(dim=1)
        if empty_rows.any():
            padding_mask[empty_rows, 0] = False

        contextual_tokens = self.context_encoder(char_tokens, src_key_padding_mask=padding_mask)
        queries = self.query_tokens.unsqueeze(0).expand(contextual_tokens.shape[0], -1, -1)
        glyph_tokens, _ = self.cross_attn(
            query=self.query_norm(queries),
            key=contextual_tokens,
            value=contextual_tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return self.output_norm(glyph_tokens)
