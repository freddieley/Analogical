"""Modality-agnostic latent core.

Architecture
------------
1. Each modality encoder produces a d_model-dim token + reliability score.
2. Tokens are reliability-weighted before fusion (soft routing).
3. A 2-layer causal transformer integrates the weighted token set into a
   shared latent state vector that accumulates across the episode.
4. The hidden GRU state provides temporal continuity between steps.

No pretrained weights. Random Kaiming initialisation only.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentCore(nn.Module):
    """Multi-modal transformer core with reliability-weighted routing.

    Parameters
    ----------
    d_model : int
        Shared embedding dimensionality (default 256).
    n_heads : int
        Transformer attention heads.
    n_layers : int
        Transformer depth.
    dropout : float
        Attention dropout (helps generalise online with few samples).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Learnable modality type embeddings (positional role, not content)
        self.modality_embeds = nn.ParameterDict()

        # Transformer that fuses all modality tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # pre-norm — more stable for online learning
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # GRU for temporal continuity across steps
        self.gru = nn.GRUCell(d_model, d_model)

        # Projection to final latent state
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        # Routing temperature (learnable)
        self.routing_temp = nn.Parameter(torch.ones(1))

        self._hidden: Optional[torch.Tensor] = None
        self._init_weights()

    # ── Registration ──────────────────────────────────────────────────────

    def register_modality(self, name: str) -> None:
        """Register a new modality by name. Creates a learnable type embedding."""
        if name not in self.modality_embeds:
            param = nn.Parameter(torch.zeros(self.d_model))
            nn.init.normal_(param, std=0.02)
            self.modality_embeds[name] = param

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        modality_tokens: Dict[str, torch.Tensor],
        reliability_scores: Dict[str, float],
        reset_hidden: bool = False,
    ) -> torch.Tensor:
        """Fuse modality tokens into a shared latent state.

        Parameters
        ----------
        modality_tokens : dict
            {modality_name: embedding_tensor of shape (d_model,)}
        reliability_scores : dict
            {modality_name: float in [0, 1]}  — automatically computed by router
        reset_hidden : bool
            True at episode start to clear GRU state.

        Returns
        -------
        latent : Tensor of shape (d_model,)
        """
        if reset_hidden:
            self._hidden = None

        if not modality_tokens:
            latent = torch.zeros(self.d_model)
            if self._hidden is None:
                self._hidden = latent
            return latent

        # Ensure all modalities are registered
        for name in modality_tokens:
            self.register_modality(name)

        # Compute routing weights (softmax over reliability scores)
        names = list(modality_tokens.keys())
        raw_scores = torch.tensor(
            [reliability_scores.get(n, 1.0) for n in names],
            dtype=torch.float32,
        )
        # Prevent all-zero weights (e.g. all sensors blacked out)
        if raw_scores.sum() < 1e-6:
            raw_scores = torch.ones_like(raw_scores)

        weights = F.softmax(raw_scores / (self.routing_temp.abs() + 1e-4), dim=0)

        # Build token sequence: (1, n_modalities, d_model)
        tokens = []
        for i, name in enumerate(names):
            tok = modality_tokens[name]                    # (d_model,)
            type_emb = self.modality_embeds[name]          # (d_model,)
            weighted = weights[i] * (tok + type_emb)
            tokens.append(weighted)

        token_seq = torch.stack(tokens, dim=0).unsqueeze(0)  # (1, n, d_model)

        # Transformer fusion
        fused = self.transformer(token_seq)                   # (1, n, d_model)
        pooled = fused.mean(dim=1).squeeze(0)                 # (d_model,)

        # GRU temporal integration
        h = self._hidden if self._hidden is not None else torch.zeros(self.d_model)
        h_new = self.gru(pooled.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        self._hidden = h_new.detach()

        latent = self.out_proj(h_new)
        return latent

    # ── Utility ───────────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Call at start of each episode to clear temporal state."""
        self._hidden = None

    @property
    def hidden_state(self) -> Optional[torch.Tensor]:
        return self._hidden

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.kaiming_uniform_(p, nonlinearity="linear")
            elif "bias" in name:
                nn.init.zeros_(p)
