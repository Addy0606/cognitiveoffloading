"""
Fine-tuned DeBERTa classifier for extraction vs. learning mode detection.

Architecture:
    DeBERTa-base → [CLS] pooling → Classification Head → Softmax
    Classification Head: Dense(768→256) → ReLU → Dropout → Dense(256→64) → ReLU → Dropout → Dense(64→2)

Classes:
    - ExtractionClassifier: Wraps the fine-tuned model for training and inference.
    - ClassificationDataset: PyTorch dataset for tokenized labeled examples.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default hyperparameters from the specification
DEFAULT_CONFIG = {
    "model_name": "microsoft/deberta-base",
    "hidden_size": 768,
    "num_labels": 2,
    "head_hidden_dims": [256, 64],
    "dropout": 0.1,
    "max_length": 256,
    "learning_rate": 2e-5,
    "batch_size": 32,
    "epochs": 3,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
}


class ClassificationDataset(Dataset):
    """
    PyTorch dataset for extraction/learning classification.

    Each item is a tokenized student message with its binary label.
    """

    def __init__(
        self,
        messages: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 256,
    ):
        """
        Args:
            messages: List of student message strings.
            labels: List of integer labels (0=learning, 1=extraction).
            tokenizer: HuggingFace tokenizer instance.
            max_length: Maximum token sequence length.
        """
        assert len(messages) == len(labels), (
            f"messages ({len(messages)}) and labels ({len(labels)}) must have same length"
        )
        self.messages = messages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        message = str(self.messages[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            message,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class ClassificationHead(nn.Module):
    """
    Custom classification head as specified:
    Dense(768→256) → ReLU → Dropout(0.1) → Dense(256→64) → ReLU → Dropout(0.1) → Dense(64→2)
    """

    def __init__(self, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class ExtractionClassifier:
    """
    Extraction/Learning classifier based on DeBERTa-base.

    Wraps HuggingFace model + custom classification head for:
    - Training on labeled data
    - Single message and batch prediction
    - Model saving and loading
    - Embedding extraction (for state encoding)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Configuration dictionary. Defaults to DEFAULT_CONFIG.
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.head = None
        self._initialized = False

    def initialize(self) -> None:
        """Load the pretrained DeBERTa model and create the classification head."""
        if self._initialized:
            return

        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading {self.config['model_name']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.model = AutoModel.from_pretrained(self.config["model_name"])
        self.head = ClassificationHead(
            hidden_size=self.config["hidden_size"],
            dropout=self.config["dropout"],
        )

        # Verify hidden size matches expectation
        actual_hidden = self.model.config.hidden_size
        if actual_hidden != self.config["hidden_size"]:
            logger.warning(
                f"Model hidden size ({actual_hidden}) differs from config "
                f"({self.config['hidden_size']}). Adjusting."
            )
            self.config["hidden_size"] = actual_hidden
            self.head = ClassificationHead(
                hidden_size=actual_hidden, dropout=self.config["dropout"]
            )

        self.model.to(self.device)
        self.head.to(self.device)
        self._initialized = True
        logger.info(
            f"Classifier initialized on {self.device} "
            f"(hidden_size={self.config['hidden_size']})"
        )

    def predict(self, message: str) -> float:
        """
        Predict extraction probability for a single message.

        Args:
            message: Student message text.

        Returns:
            Extraction probability in [0, 1].
        """
        if not message or not message.strip():
            logger.warning("Empty message passed to predict(). Returning 0.5.")
            return 0.5

        self.initialize()
        self.model.eval()
        self.head.eval()

        encoding = self.tokenizer(
            message,
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            logits = self.head(cls_embedding)
            probs = torch.softmax(logits, dim=-1)
            extraction_prob = probs[0, 1].item()  # P(extraction)

        return extraction_prob

    def predict_batch(self, messages: List[str]) -> List[float]:
        """
        Predict extraction probabilities for a batch of messages.

        Args:
            messages: List of student message strings.

        Returns:
            List of extraction probabilities in [0, 1].
        """
        if not messages:
            return []

        self.initialize()
        self.model.eval()
        self.head.eval()

        # Handle empty messages
        results = []
        valid_indices = []
        valid_messages = []
        for i, msg in enumerate(messages):
            if not msg or not str(msg).strip():
                results.append((i, 0.5))
            else:
                valid_indices.append(i)
                valid_messages.append(str(msg))

        if valid_messages:
            encoding = self.tokenizer(
                valid_messages,
                max_length=self.config["max_length"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                )
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                logits = self.head(cls_embeddings)
                probs = torch.softmax(logits, dim=-1)
                extraction_probs = probs[:, 1].cpu().tolist()

            for idx, prob in zip(valid_indices, extraction_probs):
                results.append((idx, prob))

        # Sort by original index and return just probabilities
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def get_embedding(self, message: str) -> torch.Tensor:
        """
        Extract the 768-dimensional [CLS] embedding for a message.

        Used by the state encoder to build the 771-dim state vector.

        Args:
            message: Student message text.

        Returns:
            Tensor of shape (768,) — the [CLS] token embedding.
        """
        if not message or not message.strip():
            return torch.zeros(self.config["hidden_size"])

        self.initialize()
        self.model.eval()

        encoding = self.tokenizer(
            message,
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, 768)

        return cls_embedding.squeeze(0).cpu()

    def train_model(
        self,
        train_messages: List[str],
        train_labels: List[int],
        val_messages: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        class_weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Fine-tune the classifier on labeled data.

        Args:
            train_messages: Training message texts.
            train_labels: Training labels (0=learning, 1=extraction).
            val_messages: Validation message texts (optional).
            val_labels: Validation labels (optional).
            epochs: Number of training epochs.
            learning_rate: Learning rate for AdamW.
            batch_size: Training batch size.
            class_weights: Optional class weights for imbalanced data.

        Returns:
            Dictionary with training history (losses, val metrics per epoch).
        """
        self.initialize()

        epochs = epochs or self.config["epochs"]
        learning_rate = learning_rate or self.config["learning_rate"]
        batch_size = batch_size or self.config["batch_size"]

        # Create datasets
        train_dataset = ClassificationDataset(
            train_messages, train_labels, self.tokenizer, self.config["max_length"]
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        val_loader = None
        if val_messages and val_labels:
            val_dataset = ClassificationDataset(
                val_messages, val_labels, self.tokenizer, self.config["max_length"]
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Loss function with optional class weighting
        if class_weights:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
                self.device
            )
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            # Auto-compute weights from label distribution
            label_counts = np.bincount(train_labels, minlength=2)
            if label_counts.min() > 0 and (label_counts.max() / label_counts.min() > 1.5):
                weights = len(train_labels) / (2.0 * label_counts)
                weight_tensor = torch.tensor(weights, dtype=torch.float32).to(
                    self.device
                )
                criterion = nn.CrossEntropyLoss(weight=weight_tensor)
                logger.info(f"Using class weights: {weights.tolist()}")
            else:
                criterion = nn.CrossEntropyLoss()

        # Optimizer: fine-tune last 2 layers + head
        optimizer = torch.optim.AdamW(
            [
                {"params": self.head.parameters(), "lr": learning_rate},
                {"params": self.model.parameters(), "lr": learning_rate * 0.1},
            ],
            weight_decay=self.config["weight_decay"],
        )

        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        self.model.train()
        self.head.train()

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True
            )
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                cls_emb = outputs.last_hidden_state[:, 0, :]
                logits = self.head(cls_emb)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.head.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = total_loss / max(num_batches, 1)
            history["train_loss"].append(avg_train_loss)

            # Validation
            if val_loader:
                val_loss, val_acc = self._evaluate_loader(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={avg_train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}: train_loss={avg_train_loss:.4f}"
                )

        return history

    def _evaluate_loader(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Evaluate on a DataLoader, returning (loss, accuracy)."""
        self.model.eval()
        self.head.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                cls_emb = outputs.last_hidden_state[:, 0, :]
                logits = self.head(cls_emb)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        self.model.train()
        self.head.train()
        avg_loss = total_loss / max(len(loader), 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def evaluate(
        self,
        messages: List[str],
        labels: List[int],
    ) -> Dict[str, Any]:
        """
        Evaluate classifier on a labeled dataset.

        Args:
            messages: Test message texts.
            labels: True labels (0=learning, 1=extraction).

        Returns:
            Dictionary with accuracy, precision, recall, F1,
            confusion matrix, and class distribution.
        """
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        predictions = self.predict_batch(messages)
        pred_labels = [1 if p > 0.5 else 0 for p in predictions]

        acc = accuracy_score(labels, pred_labels)
        prec = precision_score(labels, pred_labels, zero_division=0)
        rec = recall_score(labels, pred_labels, zero_division=0)
        f1 = f1_score(labels, pred_labels, zero_division=0)
        cm = confusion_matrix(labels, pred_labels)
        report = classification_report(
            labels, pred_labels, target_names=["Learning", "Extraction"]
        )

        results = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "num_samples": len(labels),
            "class_distribution": {
                "learning": int(sum(1 for l in labels if l == 0)),
                "extraction": int(sum(1 for l in labels if l == 1)),
            },
            "prediction_distribution": {
                "predicted_learning": int(sum(1 for p in pred_labels if p == 0)),
                "predicted_extraction": int(sum(1 for p in pred_labels if p == 1)),
            },
        }

        logger.info(f"\n{report}")
        return results

    def save(self, save_dir: str) -> None:
        """
        Save the classifier model and configuration.

        The DeBERTa backbone and tokenizer are saved using HuggingFace's
        save_pretrained(). The classification head is saved as a PyTorch
        state dict. The config is saved as JSON.

        Args:
            save_dir: Directory to save model artifacts.
        """
        if not self._initialized:
            raise RuntimeError("Cannot save uninitialized model. Call initialize() first.")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save backbone and tokenizer
        self.model.save_pretrained(save_path / "backbone")
        self.tokenizer.save_pretrained(save_path / "backbone")

        # Save classification head
        torch.save(self.head.state_dict(), save_path / "head.pth")

        # Save config
        with open(save_path / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Classifier saved to {save_path}")

    def load(self, load_dir: str) -> None:
        """
        Load a previously saved classifier.

        Args:
            load_dir: Directory containing saved model artifacts.
        """
        load_path = Path(load_dir)
        if not load_path.exists():
            raise FileNotFoundError(f"Model directory not found: {load_path}")

        from transformers import AutoModel, AutoTokenizer

        # Load config
        config_path = load_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)

        # Load backbone and tokenizer
        backbone_path = load_path / "backbone"
        if not backbone_path.exists():
            raise FileNotFoundError(
                f"Backbone directory not found: {backbone_path}. "
                f"Expected HuggingFace model files in {backbone_path}/"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(str(backbone_path))
        self.model = AutoModel.from_pretrained(str(backbone_path))
        self.model.to(self.device)

        # Load classification head
        head_path = load_path / "head.pth"
        if head_path.exists():
            self.head = ClassificationHead(
                hidden_size=self.config["hidden_size"],
                dropout=self.config["dropout"],
            )
            self.head.load_state_dict(
                torch.load(str(head_path), map_location=self.device, weights_only=True)
            )
            self.head.to(self.device)
        else:
            raise FileNotFoundError(
                f"Classification head not found: {head_path}"
            )

        self._initialized = True
        logger.info(f"Classifier loaded from {load_path}")

    @property
    def is_initialized(self) -> bool:
        """Check if the model is initialized and ready for inference."""
        return self._initialized
