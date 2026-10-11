"""
Shared pytest fixtures for the RL-Based Adaptive Tutoring System tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_extraction_messages():
    """Example extraction-oriented student messages (label=1)."""
    return [
        "Just give me the code for this problem",
        "What's the answer to question 3?",
        "Can you solve this for me?",
        "Tell me the solution",
        "Give me the answer",
    ]


@pytest.fixture
def sample_learning_messages():
    """Example learning-oriented student messages (label=0)."""
    return [
        "I tried using a for loop but I'm getting an index out of range error. "
        "Here's my code: for i in range(len(arr)+1). Why does this happen?",
        "I understand that recursion involves a base case, but I'm confused "
        "about how the stack unwinds. Can you explain?",
        "I attempted to implement binary search but my mid calculation seems off. "
        "I used mid = (left + right) / 2 — is there a problem with integer division?",
        "Why does quicksort have O(n²) worst case? I thought the pivot should "
        "always split evenly",
        "I think I understand how hash maps work — you hash the key to get an "
        "index — but what happens when two keys hash to the same index?",
    ]


@pytest.fixture
def random_embedding():
    """Random 768-dimensional embedding tensor."""
    return torch.randn(768)


@pytest.fixture
def random_state():
    """Random 771-dimensional state tensor."""
    return torch.randn(771)


@pytest.fixture
def dummy_transitions(random_state):
    """Generate a batch of dummy transitions for RL training."""
    transitions = []
    for _ in range(100):
        state = torch.randn(771)
        action = np.random.randint(0, 4)
        reward = np.random.uniform(-1, 1)
        next_state = torch.randn(771)
        done = np.random.random() < 0.1
        transitions.append((state, action, reward, next_state, done))
    return transitions
