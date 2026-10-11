"""
Tests for the RL Policy Network and Q-learning trainer.

Tests verify:
- PolicyNetwork architecture and output shapes
- ReplayBuffer functionality
- RLTrainer action selection (epsilon-greedy)
- Q-learning update step
- Episode training
- Model save/load roundtrip
- Training metrics tracking
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.rl_policy_network import (
    ACTION_NAMES,
    NUM_ACTIONS,
    STATE_DIM,
    PolicyNetwork,
    ReplayBuffer,
    RLTrainer,
)


# ---------------------------------------------------------------
# PolicyNetwork
# ---------------------------------------------------------------
class TestPolicyNetwork:
    def test_output_shape_single(self):
        """Single state → 4 Q-values."""
        net = PolicyNetwork()
        state = torch.randn(1, STATE_DIM)
        out = net(state)
        assert out.shape == (1, NUM_ACTIONS)

    def test_output_shape_batch(self):
        """Batch of states → batch of Q-values."""
        net = PolicyNetwork()
        state = torch.randn(8, STATE_DIM)
        out = net(state)
        assert out.shape == (8, NUM_ACTIONS)

    def test_input_dim_default(self):
        """Default input dimension is 771."""
        net = PolicyNetwork()
        state = torch.randn(1, 771)
        out = net(state)
        assert out.shape == (1, 4)

    def test_custom_dims(self):
        """Custom input/hidden/output dimensions."""
        net = PolicyNetwork(input_dim=100, hidden_dims=[64, 32], output_dim=3)
        state = torch.randn(2, 100)
        out = net(state)
        assert out.shape == (2, 3)

    def test_gradient_flow(self):
        """Gradients should flow through the network."""
        net = PolicyNetwork()
        state = torch.randn(2, STATE_DIM, requires_grad=True)
        out = net(state)
        loss = out.sum()
        loss.backward()
        assert state.grad is not None

    def test_eval_deterministic(self):
        """Eval mode should produce identical outputs."""
        net = PolicyNetwork()
        net.eval()
        state = torch.randn(2, STATE_DIM)
        out1 = net(state)
        out2 = net(state)
        assert torch.allclose(out1, out2)

    def test_parameter_count(self):
        """Network should have a reasonable parameter count."""
        net = PolicyNetwork()
        total_params = sum(p.numel() for p in net.parameters())
        # 771*256 + 256 + 256*128 + 128 + 128*64 + 64 + 64*4 + 4 ≈ 238,916
        assert total_params > 100_000
        assert total_params < 500_000

    def test_xavier_init(self):
        """Weights should be Xavier-initialized (non-zero, reasonable variance)."""
        net = PolicyNetwork()
        for m in net.modules():
            if isinstance(m, torch.nn.Linear):
                assert not torch.all(m.weight == 0)
                assert torch.all(m.bias == 0)


# ---------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------
class TestReplayBuffer:
    def test_empty_buffer(self):
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0

    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(
                torch.randn(STATE_DIM),
                action=0,
                reward=1.0,
                next_state=torch.randn(STATE_DIM),
                done=False,
            )
        assert len(buf) == 10

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(
                torch.randn(STATE_DIM), 0, 1.0, torch.randn(STATE_DIM), False
            )
        assert len(buf) == 5  # Oldest entries evicted

    def test_sample_size(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(20):
            buf.push(
                torch.randn(STATE_DIM), 0, 1.0, torch.randn(STATE_DIM), False
            )
        batch = buf.sample(5)
        assert len(batch) == 5

    def test_sample_larger_than_buffer(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(3):
            buf.push(
                torch.randn(STATE_DIM), 0, 1.0, torch.randn(STATE_DIM), False
            )
        batch = buf.sample(10)
        assert len(batch) == 3  # Clamped to buffer size

    def test_sample_structure(self):
        buf = ReplayBuffer(capacity=100)
        buf.push(torch.randn(STATE_DIM), 2, 0.5, torch.randn(STATE_DIM), True)
        batch = buf.sample(1)
        state, action, reward, next_state, done = batch[0]
        assert state.shape == (STATE_DIM,)
        assert isinstance(action, int)
        assert isinstance(reward, float)
        assert next_state.shape == (STATE_DIM,)
        assert isinstance(done, bool)


# ---------------------------------------------------------------
# RLTrainer
# ---------------------------------------------------------------
class TestRLTrainer:
    def test_initialization(self):
        trainer = RLTrainer()
        assert trainer.epsilon == 0.2
        assert trainer.gamma == 0.99
        assert trainer.episode_count == 0

    def test_select_action_returns_valid_id(self, random_state):
        trainer = RLTrainer()
        for _ in range(20):
            action = trainer.select_action(random_state)
            assert 0 <= action < NUM_ACTIONS

    def test_select_action_greedy(self, random_state):
        """Greedy mode should always pick the same action."""
        trainer = RLTrainer()
        actions = [trainer.select_action(random_state, greedy=True) for _ in range(10)]
        assert len(set(actions)) == 1  # All same action

    def test_select_action_full_exploration(self, random_state):
        """With epsilon=1.0, all actions should appear (probabilistically)."""
        trainer = RLTrainer(epsilon=1.0)
        actions = set()
        for _ in range(200):
            actions.add(trainer.select_action(random_state, epsilon=1.0))
        assert len(actions) == NUM_ACTIONS

    def test_get_q_values_shape(self, random_state):
        trainer = RLTrainer()
        q_values = trainer.get_q_values(random_state)
        assert len(q_values) == NUM_ACTIONS
        assert all(isinstance(q, float) for q in q_values)

    def test_train_step_returns_loss(self, random_state):
        trainer = RLTrainer(batch_size=4)
        # Fill buffer first
        for _ in range(10):
            trainer.replay_buffer.push(
                torch.randn(STATE_DIM), 0, 0.5, torch.randn(STATE_DIM), False
            )
        loss = trainer.train_step(
            random_state, 1, 0.5, torch.randn(STATE_DIM), False
        )
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_train_step_insufficient_buffer(self, random_state):
        """Should return 0.0 when buffer is too small."""
        trainer = RLTrainer(batch_size=64)
        loss = trainer.train_step(
            random_state, 0, 1.0, torch.randn(STATE_DIM), False
        )
        assert loss == 0.0

    def test_train_episode(self, dummy_transitions):
        trainer = RLTrainer(batch_size=16)
        metrics = trainer.train_episode(dummy_transitions)
        assert "total_reward" in metrics
        assert "avg_loss" in metrics
        assert "num_steps" in metrics
        assert "epsilon" in metrics
        assert "episode" in metrics
        assert metrics["num_steps"] == len(dummy_transitions)
        assert trainer.episode_count == 1

    def test_epsilon_decay(self, dummy_transitions):
        trainer = RLTrainer(epsilon=0.5, epsilon_decay=0.9, batch_size=16)
        initial_eps = trainer.epsilon
        trainer.train_episode(dummy_transitions)
        assert trainer.epsilon < initial_eps
        assert trainer.epsilon >= trainer.epsilon_min

    def test_save_and_load(self, dummy_transitions):
        trainer = RLTrainer(batch_size=16)
        trainer.train_episode(dummy_transitions)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "policy.pth")
            trainer.save(path)
            assert Path(path).exists()

            # Load into new trainer
            new_trainer = RLTrainer()
            new_trainer.load(path)
            assert new_trainer.episode_count == trainer.episode_count

            # Q-values should match
            state = torch.randn(STATE_DIM)
            q1 = trainer.get_q_values(state)
            q2 = new_trainer.get_q_values(state)
            for a, b in zip(q1, q2):
                assert abs(a - b) < 1e-5

    def test_load_nonexistent_raises(self):
        trainer = RLTrainer()
        with pytest.raises(FileNotFoundError):
            trainer.load("/nonexistent/path.pth")

    def test_get_training_metrics_empty(self):
        trainer = RLTrainer()
        metrics = trainer.get_training_metrics()
        assert metrics["status"] == "no training data"

    def test_get_training_metrics_after_training(self, dummy_transitions):
        trainer = RLTrainer(batch_size=16)
        trainer.train_episode(dummy_transitions)
        metrics = trainer.get_training_metrics()
        assert "total_episodes" in metrics
        assert "avg_reward" in metrics
        assert metrics["total_episodes"] == 1

    def test_target_network_update(self, dummy_transitions):
        """Target network should update after target_update_freq episodes."""
        trainer = RLTrainer(
            target_update_freq=2, batch_size=16
        )
        # Get initial target network params
        initial_params = {
            k: v.clone()
            for k, v in trainer.target_net.state_dict().items()
        }

        # Train one episode — target should NOT update
        trainer.train_episode(dummy_transitions)
        for k, v in trainer.target_net.state_dict().items():
            assert torch.allclose(v, initial_params[k])

        # Train second episode — target SHOULD update
        trainer.train_episode(dummy_transitions)
        # After update, target should match policy
        for k in trainer.policy_net.state_dict():
            assert torch.allclose(
                trainer.target_net.state_dict()[k],
                trainer.policy_net.state_dict()[k],
            )


# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
class TestConstants:
    def test_action_names(self):
        assert ACTION_NAMES == {
            0: "REDIRECT",
            1: "SCAFFOLD",
            2: "SOCRATIC",
            3: "DELAY",
        }

    def test_num_actions(self):
        assert NUM_ACTIONS == 4

    def test_state_dim(self):
        assert STATE_DIM == 771
