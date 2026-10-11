"""
Tests for the state encoder utility function.

Tests verify:
- Output dimension is exactly 771
- L2-normalization of the embedding component
- Scalar features are appended correctly
- Turn number normalization
- Edge cases (zero embedding, extreme values)
"""

import pytest
import torch

from src.utils import EMBEDDING_DIM, STATE_DIM, encode_state


class TestEncodeState:
    def test_output_shape(self, random_embedding):
        state = encode_state(
            message_embedding=random_embedding,
            extraction_score=0.7,
            engagement_score=0.5,
            turn_number=3,
        )
        assert state.shape == (STATE_DIM,)
        assert state.shape == (771,)

    def test_output_dtype(self, random_embedding):
        state = encode_state(
            message_embedding=random_embedding,
            extraction_score=0.5,
            engagement_score=0.5,
            turn_number=1,
        )
        assert state.dtype == torch.float32

    def test_embedding_l2_normalized(self, random_embedding):
        """The first 768 dims should be L2-normalized."""
        state = encode_state(
            message_embedding=random_embedding,
            extraction_score=0.5,
            engagement_score=0.5,
            turn_number=1,
        )
        embedding_part = state[:EMBEDDING_DIM]
        norm = torch.norm(embedding_part, p=2)
        assert abs(norm.item() - 1.0) < 1e-5, (
            f"Embedding norm should be 1.0, got {norm.item()}"
        )

    def test_scalar_features_position(self, random_embedding):
        """Scalar features at positions 768, 769, 770."""
        state = encode_state(
            message_embedding=random_embedding,
            extraction_score=0.8,
            engagement_score=0.3,
            turn_number=10,
            max_turns=50,
        )
        assert abs(state[768].item() - 0.8) < 1e-5  # extraction_score
        assert abs(state[769].item() - 0.3) < 1e-5  # engagement_score
        assert abs(state[770].item() - 0.2) < 1e-5  # 10/50 = 0.2

    def test_turn_number_normalization(self, random_embedding):
        """Turn number should be normalized to [0, 1]."""
        state = encode_state(
            message_embedding=random_embedding,
            extraction_score=0.5,
            engagement_score=0.5,
            turn_number=25,
            max_turns=50,
        )
        assert abs(state[770].item() - 0.5) < 1e-5

    def test_turn_number_capped_at_1(self, random_embedding):
        """Turn numbers beyond max_turns should be capped at 1.0."""
        state = encode_state(
            message_embedding=random_embedding,
            extraction_score=0.5,
            engagement_score=0.5,
            turn_number=100,
            max_turns=50,
        )
        assert abs(state[770].item() - 1.0) < 1e-5

    def test_zero_embedding(self):
        """Zero embedding should still produce correct dimension."""
        zero_emb = torch.zeros(EMBEDDING_DIM)
        state = encode_state(
            message_embedding=zero_emb,
            extraction_score=0.5,
            engagement_score=0.5,
            turn_number=1,
        )
        assert state.shape == (STATE_DIM,)
        # Zero embedding stays zero (no normalization possible)
        assert torch.all(state[:EMBEDDING_DIM] == 0.0)

    def test_2d_embedding_squeezed(self):
        """A (1, 768) embedding should be squeezed to (768,)."""
        emb_2d = torch.randn(1, EMBEDDING_DIM)
        state = encode_state(
            message_embedding=emb_2d,
            extraction_score=0.5,
            engagement_score=0.5,
            turn_number=1,
        )
        assert state.shape == (STATE_DIM,)

    def test_wrong_embedding_dim_raises(self):
        """Embedding with wrong dimension should raise."""
        bad_emb = torch.randn(512)
        with pytest.raises(AssertionError):
            encode_state(
                message_embedding=bad_emb,
                extraction_score=0.5,
                engagement_score=0.5,
                turn_number=1,
            )

    def test_boundary_scores(self, random_embedding):
        """Edge values 0.0 and 1.0 should work."""
        for ext, eng in [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]:
            state = encode_state(
                message_embedding=random_embedding,
                extraction_score=ext,
                engagement_score=eng,
                turn_number=1,
            )
            assert state.shape == (STATE_DIM,)
            assert abs(state[768].item() - ext) < 1e-5
            assert abs(state[769].item() - eng) < 1e-5

    def test_reproducibility(self, random_embedding):
        """Same inputs should produce identical outputs."""
        args = dict(
            message_embedding=random_embedding,
            extraction_score=0.7,
            engagement_score=0.3,
            turn_number=5,
        )
        s1 = encode_state(**args)
        s2 = encode_state(**args)
        assert torch.allclose(s1, s2)
