"""
Tests for the DeBERTa extraction classifier module.

Tests that DON'T require model download:
- ClassificationHead architecture and shapes
- ClassificationDataset structure
- ExtractionClassifier configuration
- Empty message handling

Tests that require model download are marked with @pytest.mark.slow
and can be run separately.
"""

import pytest
import torch
import torch.nn as nn

from src.classifier import (
    ClassificationDataset,
    ClassificationHead,
    DEFAULT_CONFIG,
    ExtractionClassifier,
)


# ---------------------------------------------------------------
# ClassificationHead
# ---------------------------------------------------------------
class TestClassificationHead:
    def test_architecture(self):
        """Head should have the specified layer structure."""
        head = ClassificationHead(hidden_size=768, dropout=0.1)
        # Check it's an nn.Module
        assert isinstance(head, nn.Module)

    def test_output_shape(self):
        """Input (batch, 768) → output (batch, 2)."""
        head = ClassificationHead(hidden_size=768, dropout=0.1)
        x = torch.randn(4, 768)
        out = head(x)
        assert out.shape == (4, 2)

    def test_single_sample(self):
        """Input (1, 768) → output (1, 2)."""
        head = ClassificationHead(hidden_size=768, dropout=0.1)
        x = torch.randn(1, 768)
        out = head(x)
        assert out.shape == (1, 2)

    def test_different_hidden_size(self):
        """Should work with non-default hidden sizes."""
        head = ClassificationHead(hidden_size=256, dropout=0.1)
        x = torch.randn(2, 256)
        out = head(x)
        assert out.shape == (2, 2)

    def test_gradient_flow(self):
        """Gradients should flow through the head."""
        head = ClassificationHead(hidden_size=768, dropout=0.1)
        head.train()
        x = torch.randn(2, 768, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_eval_mode_deterministic(self):
        """In eval mode, outputs should be deterministic."""
        head = ClassificationHead(hidden_size=768, dropout=0.1)
        head.eval()
        x = torch.randn(2, 768)
        out1 = head(x)
        out2 = head(x)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------
# ExtractionClassifier (config only, no model download)
# ---------------------------------------------------------------
class TestExtractionClassifierConfig:
    def test_default_config(self):
        clf = ExtractionClassifier()
        assert clf.config["model_name"] == "microsoft/deberta-base"
        assert clf.config["hidden_size"] == 768
        assert clf.config["num_labels"] == 2
        assert clf.config["max_length"] == 256

    def test_custom_config(self):
        custom = {"max_length": 128, "epochs": 5}
        clf = ExtractionClassifier(config=custom)
        assert clf.config["max_length"] == 128
        assert clf.config["epochs"] == 5
        # Defaults still present
        assert clf.config["hidden_size"] == 768

    def test_not_initialized_before_call(self):
        clf = ExtractionClassifier()
        assert not clf.is_initialized

    def test_predict_empty_message(self):
        """Empty message returns 0.5 without initializing the model."""
        clf = ExtractionClassifier()
        assert clf.predict("") == 0.5
        assert clf.predict("   ") == 0.5
        assert not clf.is_initialized  # Should not have loaded the model

    def test_predict_batch_empty(self):
        clf = ExtractionClassifier()
        assert clf.predict_batch([]) == []

    def test_save_uninitialized_raises(self):
        clf = ExtractionClassifier()
        with pytest.raises(RuntimeError):
            clf.save("/tmp/test_model")

    def test_load_missing_directory_raises(self):
        clf = ExtractionClassifier()
        with pytest.raises(FileNotFoundError):
            clf.load("/nonexistent/path/to/model")


# ---------------------------------------------------------------
# ClassificationDataset
# ---------------------------------------------------------------
class TestClassificationDataset:
    def _make_mock_tokenizer(self):
        """Create a minimal mock tokenizer for testing."""

        class MockTokenizer:
            def __call__(self, text, max_length=256, padding="max_length",
                         truncation=True, return_tensors="pt"):
                # Return dummy tensors
                return {
                    "input_ids": torch.ones(1, max_length, dtype=torch.long),
                    "attention_mask": torch.ones(1, max_length, dtype=torch.long),
                }

        return MockTokenizer()

    def test_length(self):
        tokenizer = self._make_mock_tokenizer()
        ds = ClassificationDataset(
            messages=["msg1", "msg2", "msg3"],
            labels=[0, 1, 0],
            tokenizer=tokenizer,
        )
        assert len(ds) == 3

    def test_getitem_keys(self):
        tokenizer = self._make_mock_tokenizer()
        ds = ClassificationDataset(
            messages=["hello"],
            labels=[1],
            tokenizer=tokenizer,
        )
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "label" in item

    def test_getitem_shapes(self):
        tokenizer = self._make_mock_tokenizer()
        ds = ClassificationDataset(
            messages=["hello"],
            labels=[1],
            tokenizer=tokenizer,
            max_length=64,
        )
        item = ds[0]
        assert item["input_ids"].shape == (64,)
        assert item["attention_mask"].shape == (64,)
        assert item["label"].shape == ()

    def test_label_dtype(self):
        tokenizer = self._make_mock_tokenizer()
        ds = ClassificationDataset(
            messages=["hello"],
            labels=[1],
            tokenizer=tokenizer,
        )
        item = ds[0]
        assert item["label"].dtype == torch.long

    def test_mismatched_lengths_raises(self):
        tokenizer = self._make_mock_tokenizer()
        with pytest.raises(AssertionError):
            ClassificationDataset(
                messages=["a", "b"],
                labels=[0],
                tokenizer=tokenizer,
            )


# ---------------------------------------------------------------
# Slow tests requiring model download
# ---------------------------------------------------------------
@pytest.mark.slow
class TestExtractionClassifierWithModel:
    """These tests download the DeBERTa model. Run with: pytest -m slow"""

    def test_initialize(self):
        clf = ExtractionClassifier()
        clf.initialize()
        assert clf.is_initialized
        assert clf.model is not None
        assert clf.tokenizer is not None
        assert clf.head is not None

    def test_predict_returns_float(self):
        clf = ExtractionClassifier()
        score = clf.predict("Just give me the answer")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_predict_batch(self):
        clf = ExtractionClassifier()
        scores = clf.predict_batch(["Hello", "Give me the code"])
        assert len(scores) == 2
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_get_embedding_shape(self):
        clf = ExtractionClassifier()
        emb = clf.get_embedding("Test message")
        assert emb.shape == (768,)
        assert emb.dtype == torch.float32

    def test_get_embedding_empty(self):
        clf = ExtractionClassifier()
        emb = clf.get_embedding("")
        assert emb.shape == (768,)
        assert torch.all(emb == 0.0)
