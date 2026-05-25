"""Neural memory bank with frequency-weighted associative recall.

Motivation
----------
Human procedural memory strengthens with repetition: actions performed
repeatedly become automatic and are recalled with higher fidelity.  This
module implements a differentiable analogue:

  - Each memory *slot* stores a (key_latent, action, value) triple.
  - Every time a slot is written or retrieved its *frequency counter* increments.
  - Retrieval attention = softmax(cosine_sim(query, key) * freq_weight / τ)
    where freq_weight = log(1 + frequency).
  - Eviction policy: when capacity is full, the slot with the lowest
    *frequency-weighted value* (freq * value) is overwritten — high-frequency
    useful memories survive indefinitely.

Integration with MPPI
---------------------
``query()`` returns the most-recalled successful action for the current latent
context.  ``MPPIPlanner`` receives this as a *warm-start prior* so that
frequently-used good actions bias the trajectory search without overriding it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class _MemorySlot:
    key: torch.Tensor     # (d_model,)
    action: torch.Tensor  # (action_dim,)
    value: float          # estimated return — EMA-updated on revisit
    frequency: int        # visit count


class NeuralMemoryBank:
    """Capacity-limited associative memory with frequency-weighted recall.

    Parameters
    ----------
    capacity   : maximum number of distinct memories stored
    d_model    : key embedding dimensionality
    action_dim : action dimensionality
    temperature: retrieval softmax temperature τ (lower → more peaked)
    sim_thresh : cosine similarity above which a query is counted as a hit
                 (updates frequency of the closest matching slot)
    value_lr   : EMA coefficient for updating slot values on revisit
    replay_k   : top-K high-frequency slots replayed on each consolidation
    """

    def __init__(
        self,
        capacity: int = 1024,
        d_model: int = 256,
        action_dim: int = 1,
        temperature: float = 0.1,
        sim_thresh: float = 0.85,
        value_lr: float = 0.1,
        replay_k: int = 16,
    ) -> None:
        self.capacity = capacity
        self.d_model = d_model
        self.action_dim = action_dim
        self.temperature = temperature
        self.sim_thresh = sim_thresh
        self.value_lr = value_lr
        self.replay_k = replay_k

        # Pre-allocated tensors (faster than list of dataclasses for large banks)
        self._keys = torch.zeros(capacity, d_model)       # normalised latent keys
        self._actions = torch.zeros(capacity, action_dim)
        self._values = torch.zeros(capacity)              # expected returns
        self._freqs = torch.zeros(capacity, dtype=torch.int64)
        self._size: int = 0   # number of valid slots

    # ── Write / update ────────────────────────────────────────────────────

    def _to_cpu_float(self, t: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is float32 on CPU (all memory ops run on CPU)."""
        return t.float().cpu()

    def write(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        value: float,
    ) -> None:
        """Store or update a (latent, action, value) memory.

        If a slot with cosine similarity ≥ sim_thresh exists, update it
        (increment frequency, EMA-update value).  Otherwise allocate a new
        slot, evicting the least-valuable if at capacity.
        """
        with torch.no_grad():
            key = F.normalize(self._to_cpu_float(latent), dim=0)
            act = self._to_cpu_float(action)

            if self._size > 0:
                sims = F.cosine_similarity(
                    key.unsqueeze(0), self._keys[: self._size], dim=1
                )
                best_idx = int(sims.argmax().item())
                if sims[best_idx].item() >= self.sim_thresh:
                    # Update existing slot
                    self._values[best_idx] = (
                        (1 - self.value_lr) * self._values[best_idx]
                        + self.value_lr * value
                    )
                    self._freqs[best_idx] += 1
                    # Blend action towards most recently taken
                    self._actions[best_idx] = (
                        0.9 * self._actions[best_idx] + 0.1 * act
                    )
                    return

            # New slot needed
            if self._size < self.capacity:
                idx = self._size
                self._size += 1
            else:
                # Evict slot with lowest frequency-weighted value
                scores = self._freqs.float() * self._values
                idx = int(scores.argmin().item())

            self._keys[idx] = key
            self._actions[idx] = act
            self._values[idx] = value
            self._freqs[idx] = 1

    # ── Read / recall ─────────────────────────────────────────────────────

    def query(
        self, latent: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], float, float]:
        """Retrieve the best action suggestion for the given latent context.

        Returns
        -------
        action_prior : (action_dim,) tensor — frequency-weighted best action,
                       or None if the bank is empty.
        expected_value : float — recall-weighted expected return.
        confidence  : float in [0, 1] — how certain the recall is
                      (peak attention weight × normalised frequency of top slot).
        """
        if self._size == 0:
            return None, 0.0, 0.0

        with torch.no_grad():
            key = F.normalize(self._to_cpu_float(latent), dim=0)
            n = self._size
            sims = F.cosine_similarity(key.unsqueeze(0), self._keys[:n], dim=1)

            # Frequency boost: log(1 + freq)
            freq_weight = torch.log1p(self._freqs[:n].float())

            # Joint score: similarity scaled by frequency (temperature-softmax)
            raw_scores = sims * freq_weight / (self.temperature + 1e-8)
            attn = F.softmax(raw_scores, dim=0)  # (n,)

            # Weighted action
            action_prior = (attn.unsqueeze(1) * self._actions[:n]).sum(dim=0)

            # Expected value
            expected_value = float((attn * self._values[:n]).sum().item())

            # Confidence: max attention × normalised top-freq
            top_idx = int(attn.argmax().item())
            max_freq = float(self._freqs[:n].max().item())
            freq_norm = (
                float(self._freqs[top_idx].item()) / (max_freq + 1e-8)
            )
            confidence = float(attn[top_idx].item()) * freq_norm

        return action_prior, expected_value, confidence

    # ── Memory consolidation (replay) ──────────────────────────────────────

    def top_k_memories(
        self, k: Optional[int] = None
    ) -> list[tuple[torch.Tensor, torch.Tensor, float, int]]:
        """Return the top-k most frequent memories for replay consolidation.

        Each entry: (key, action, value, frequency).
        """
        k = k or self.replay_k
        if self._size == 0:
            return []
        n = self._size
        topk_idx = self._freqs[:n].topk(min(k, n)).indices
        return [
            (
                self._keys[i].clone(),
                self._actions[i].clone(),
                float(self._values[i].item()),
                int(self._freqs[i].item()),
            )
            for i in topk_idx
        ]

    # ── Episode management ────────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Clear all memories at episode start (no carryover between episodes).

        Frequency information from prior episodes is NOT preserved — the agent
        builds its memory from scratch each episode, mirroring the
        zero-pretraining constraint.
        """
        self._size = 0
        self._keys.zero_()
        self._actions.zero_()
        self._values.zero_()
        self._freqs.zero_()

    # ── Summary ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        if self._size == 0:
            return {"size": 0, "max_freq": 0, "mean_value": 0.0}
        n = self._size
        return {
            "size": n,
            "capacity": self.capacity,
            "max_freq": int(self._freqs[:n].max().item()),
            "mean_freq": round(float(self._freqs[:n].float().mean().item()), 2),
            "mean_value": round(float(self._values[:n].mean().item()), 4),
        }
