"""
Tests for utility functions in src/utils.py.

Tests verify:
- Data loading and validation
- Synthetic data generation (shape, columns, labels)
- Reward calculation
- Tutor response categorization
- LLM fallback responses
- JSON save/load helpers
- Seed setting
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.utils import (
    ACTION_NAMES,
    EMBEDDING_DIM,
    STATE_DIM,
    calculate_reward,
    call_llm,
    categorize_tutor_response,
    generate_synthetic_dataset,
    generate_synthetic_labeled,
    get_llm_provider,
    load_csv,
    load_json,
    save_json,
    set_seed,
)


# ---------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------
class TestSetSeed:
    def test_reproducibility(self):
        set_seed(42)
        a1 = np.random.rand(5)
        t1 = torch.rand(5)

        set_seed(42)
        a2 = np.random.rand(5)
        t2 = torch.rand(5)

        np.testing.assert_array_equal(a1, a2)
        assert torch.allclose(t1, t2)


# ---------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------
class TestLoadCSV:
    def test_load_valid_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        df.to_csv(csv_path, index=False)

        loaded = load_csv(str(csv_path))
        assert len(loaded) == 2
        assert "col1" in loaded.columns

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/file.csv")

    def test_missing_columns_raises(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"col1": [1]})
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Missing required columns"):
            load_csv(str(csv_path), required_columns={"col1", "col2"})

    def test_required_columns_present(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"col1": [1], "col2": ["a"]})
        df.to_csv(csv_path, index=False)

        loaded = load_csv(str(csv_path), required_columns={"col1", "col2"})
        assert len(loaded) == 1


# ---------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------
class TestGenerateSyntheticDataset:
    def test_output_type(self):
        df = generate_synthetic_dataset(num_dialogues=5)
        assert isinstance(df, pd.DataFrame)

    def test_columns(self):
        df = generate_synthetic_dataset(num_dialogues=5)
        expected = {"dialogue_id", "turn_num", "speaker", "message", "context"}
        assert set(df.columns) == expected

    def test_num_dialogues(self):
        df = generate_synthetic_dataset(num_dialogues=10)
        assert df["dialogue_id"].nunique() == 10

    def test_dialogue_ids_prefixed(self):
        df = generate_synthetic_dataset(num_dialogues=3)
        for did in df["dialogue_id"].unique():
            assert did.startswith("SYNTH_")

    def test_speakers_valid(self):
        df = generate_synthetic_dataset(num_dialogues=5)
        assert set(df["speaker"].unique()) == {"student", "tutor"}

    def test_turn_numbers_start_at_1(self):
        df = generate_synthetic_dataset(num_dialogues=5)
        for _, group in df.groupby("dialogue_id"):
            assert group["turn_num"].min() == 1

    def test_reproducibility(self):
        df1 = generate_synthetic_dataset(num_dialogues=5, seed=42)
        df2 = generate_synthetic_dataset(num_dialogues=5, seed=42)
        pd.testing.assert_frame_equal(df1, df2)


class TestGenerateSyntheticLabeled:
    def test_output_type(self):
        df = generate_synthetic_labeled(num_samples=50)
        assert isinstance(df, pd.DataFrame)

    def test_columns(self):
        df = generate_synthetic_labeled(num_samples=50)
        expected = {"dialogue_id", "turn_num", "message", "label", "confidence", "notes"}
        assert set(df.columns) == expected

    def test_num_samples(self):
        df = generate_synthetic_labeled(num_samples=100)
        assert len(df) == 100

    def test_label_values(self):
        df = generate_synthetic_labeled(num_samples=100)
        assert set(df["label"].unique()) == {0, 1}

    def test_extraction_ratio(self):
        df = generate_synthetic_labeled(num_samples=1000, extraction_ratio=0.3)
        actual_ratio = df["label"].mean()
        assert abs(actual_ratio - 0.3) < 0.05

    def test_synthetic_notes(self):
        df = generate_synthetic_labeled(num_samples=50)
        valid_notes = {"synthetic_extraction", "synthetic_learning"}
        assert set(df["notes"].unique()) == valid_notes

    def test_reproducibility(self):
        df1 = generate_synthetic_labeled(num_samples=50, seed=42)
        df2 = generate_synthetic_labeled(num_samples=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------
# Reward Calculation
# ---------------------------------------------------------------
class TestCalculateReward:
    def test_positive_reward(self):
        """Student moved from extraction to learning → positive reward."""
        reward = calculate_reward(0.8, 0.3)
        assert reward == pytest.approx(0.5, abs=1e-5)

    def test_negative_reward(self):
        """Student moved from learning to extraction → negative reward."""
        reward = calculate_reward(0.2, 0.7)
        assert reward == pytest.approx(-0.5, abs=1e-5)

    def test_zero_reward(self):
        """No change → zero reward."""
        reward = calculate_reward(0.5, 0.5)
        assert reward == pytest.approx(0.0, abs=1e-5)

    def test_return_type(self):
        reward = calculate_reward(0.5, 0.3)
        assert isinstance(reward, float)

    def test_extreme_values(self):
        assert calculate_reward(1.0, 0.0) == pytest.approx(1.0)
        assert calculate_reward(0.0, 1.0) == pytest.approx(-1.0)


# ---------------------------------------------------------------
# Tutor Response Categorization
# ---------------------------------------------------------------
class TestCategorizeTutorResponse:
    def test_redirect(self):
        msg = "What have you tried so far? Show me your attempt first."
        assert categorize_tutor_response(msg) == 0  # REDIRECT

    def test_scaffold(self):
        msg = "Good start! Here's a hint to help you move forward."
        assert categorize_tutor_response(msg) == 1  # SCAFFOLD

    def test_socratic(self):
        msg = "What do you think would happen if you tried a smaller input first?"
        assert categorize_tutor_response(msg) == 2  # SOCRATIC

    def test_delay(self):
        msg = "You've been working hard. Take a break and come back fresh."
        assert categorize_tutor_response(msg) == 3  # DELAY

    def test_empty_returns_scaffold(self):
        assert categorize_tutor_response("") == 1  # Default

    def test_none_returns_scaffold(self):
        assert categorize_tutor_response(None) == 1  # Default

    def test_direct_answer_defaults_to_scaffold(self):
        msg = "Here's the solution: use a dictionary to store the values."
        result = categorize_tutor_response(msg)
        assert result == 1  # Direct is categorized as SCAFFOLD

    def test_all_return_valid_action_ids(self):
        messages = [
            "What have you tried?",
            "Here's a hint",
            "What would happen if you changed X?",
            "Take a break",
            "",
            "Generic response",
        ]
        for msg in messages:
            action = categorize_tutor_response(msg)
            assert 0 <= action <= 3
            assert action in ACTION_NAMES


# ---------------------------------------------------------------
# LLM Integration
# ---------------------------------------------------------------
class TestLLMFallback:
    def test_fallback_redirect(self):
        response = call_llm(
            "You are a redirect tutor. Redirect the student.",
            "Give me the answer",
            provider="fallback",
        )
        assert "FALLBACK" in response

    def test_fallback_scaffold(self):
        response = call_llm(
            "You are a scaffold tutor.",
            "Help me",
            provider="fallback",
        )
        assert "FALLBACK" in response

    def test_fallback_socratic(self):
        response = call_llm(
            "Use the socratic method.",
            "I'm stuck",
            provider="fallback",
        )
        assert "FALLBACK" in response

    def test_fallback_delay(self):
        response = call_llm(
            "Suggest a delay or break.",
            "I'm frustrated",
            provider="fallback",
        )
        assert "FALLBACK" in response

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            call_llm("system", "user", provider="invalid_provider")


class TestGetLLMProvider:
    def test_returns_string(self):
        provider = get_llm_provider()
        assert isinstance(provider, str)
        assert provider in ["openai", "fallback", "ollama"]


# ---------------------------------------------------------------
# JSON Helpers
# ---------------------------------------------------------------
class TestJSONHelpers:
    def test_save_and_load(self, tmp_path):
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        path = str(tmp_path / "test.json")
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_save_creates_directories(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "test.json")
        save_json({"test": True}, path)
        assert Path(path).exists()

    def test_handles_non_serializable(self, tmp_path):
        """save_json should handle numpy/torch types via default=str."""
        path = str(tmp_path / "test.json")
        data = {"tensor_shape": torch.Size([2, 3])}
        save_json(data, path)  # Should not raise
        assert Path(path).exists()


# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
class TestConstants:
    def test_state_dim(self):
        assert STATE_DIM == 771

    def test_embedding_dim(self):
        assert EMBEDDING_DIM == 768

    def test_action_names(self):
        assert len(ACTION_NAMES) == 4
        assert ACTION_NAMES[0] == "REDIRECT"
        assert ACTION_NAMES[1] == "SCAFFOLD"
        assert ACTION_NAMES[2] == "SOCRATIC"
        assert ACTION_NAMES[3] == "DELAY"
