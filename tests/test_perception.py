"""Tests for perception encoders and latent core."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from analogical.perception.encoders import (
    AudioEncoder,
    ProprioceptionEncoder,
    TactileEncoder,
    VisionEncoder,
)
from analogical.perception.latent_core import LatentCore


# ── Encoder tests ────────────────────────────────────────────────────────

class TestProprioceptionEncoder:
    def test_output_shape(self):
        enc = ProprioceptionEncoder(input_dim=4, d_model=64)
        x = torch.randn(4)
        emb, rel = enc(x)
        assert emb.shape == (64,)
        assert float(rel) == 1.0

    def test_none_input(self):
        enc = ProprioceptionEncoder(input_dim=4, d_model=64)
        emb, rel = enc(None)
        assert emb.shape == (64,)
        assert torch.all(emb == 0.0)
        assert float(rel) == 0.0

    def test_no_pretrained_weights(self):
        enc = ProprioceptionEncoder(input_dim=4, d_model=64)
        # Check all biases start at zero
        for name, p in enc.named_parameters():
            if "bias" in name:
                assert torch.all(p == 0.0), f"Bias {name} not zero-initialised"


class TestVisionEncoder:
    def test_output_shape(self):
        enc = VisionEncoder(d_model=128)
        img = torch.randn(1, 64, 64)
        emb, rel = enc(img)
        assert emb.shape == (128,)
        assert float(rel) == 1.0

    def test_none_input(self):
        enc = VisionEncoder(d_model=128)
        emb, rel = enc(None)
        assert emb.shape == (128,)
        assert float(rel) == 0.0

    def test_batched_input(self):
        enc = VisionEncoder(d_model=128)
        img = torch.randn(1, 1, 64, 64)  # (B, C, H, W)
        emb, rel = enc(img)
        assert emb.shape == (128,)


class TestTactileEncoder:
    def test_output_shape(self):
        enc = TactileEncoder(input_dim=1, d_model=64)
        x = torch.tensor([0.5])
        emb, rel = enc(x)
        assert emb.shape == (64,)

    def test_none_input(self):
        enc = TactileEncoder(d_model=64)
        emb, rel = enc(None)
        assert float(rel) == 0.0


class TestAudioEncoder:
    def test_output_shape(self):
        enc = AudioEncoder(d_model=64)
        x = torch.randn(512)
        emb, rel = enc(x)
        assert emb.shape == (64,)

    def test_none_input(self):
        enc = AudioEncoder(d_model=64)
        emb, rel = enc(None)
        assert float(rel) == 0.0


# ── LatentCore tests ─────────────────────────────────────────────────────

class TestLatentCore:
    def _make_core(self):
        return LatentCore(d_model=64, n_heads=2, n_layers=1)

    def test_output_shape(self):
        core = self._make_core()
        tokens = {"proprioception": torch.randn(64), "vision": torch.randn(64)}
        reliability = {"proprioception": 0.9, "vision": 0.5}
        latent = core(tokens, reliability)
        assert latent.shape == (64,)

    def test_single_modality(self):
        core = self._make_core()
        tokens = {"proprioception": torch.randn(64)}
        reliability = {"proprioception": 1.0}
        latent = core(tokens, reliability)
        assert latent.shape == (64,)

    def test_empty_tokens(self):
        core = self._make_core()
        latent = core({}, {})
        assert latent.shape == (64,)
        assert torch.all(latent == 0.0)

    def test_hidden_state_accumulates(self):
        core = self._make_core()
        tokens = {"proprioception": torch.randn(64)}
        latent1 = core(tokens, {"proprioception": 1.0})
        latent2 = core(tokens, {"proprioception": 1.0})
        # Hidden state should change the output
        assert not torch.allclose(latent1, latent2)

    def test_reset_clears_hidden(self):
        core = self._make_core()
        core.eval()   # disable dropout for deterministic comparison
        tokens = {"proprioception": torch.randn(64)}
        with torch.no_grad():
            latent1 = core(tokens, {"proprioception": 1.0}, reset_hidden=True)
            # After reset, same input should give same output (deterministic forward)
            latent2 = core(tokens, {"proprioception": 1.0}, reset_hidden=True)
        assert torch.allclose(latent1, latent2, atol=1e-5)

    def test_zero_reliability_all_modalities(self):
        """All-zero reliability should not crash — uniform fallback used."""
        core = self._make_core()
        tokens = {"proprioception": torch.randn(64)}
        latent = core(tokens, {"proprioception": 0.0})
        assert latent.shape == (64,)
        assert torch.isfinite(latent).all()

    def test_modality_registration_on_new_name(self):
        core = self._make_core()
        tokens = {"new_sensor": torch.randn(64)}
        latent = core(tokens, {"new_sensor": 1.0})
        assert "new_sensor" in core.modality_embeds

    def test_no_pretrained_weights(self):
        core = self._make_core()
        for name, p in core.named_parameters():
            assert torch.isfinite(p).all(), f"Non-finite parameter: {name}"
