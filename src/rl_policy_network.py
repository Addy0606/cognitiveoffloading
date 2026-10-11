"""
RL Policy Network and Q-learning trainer.

Architecture:
    Input (771) → Dense(256) → ReLU → Dense(128) → ReLU → Dense(64) → ReLU → Dense(4)
    Output: 4 raw Q-values (one per action: REDIRECT, SCAFFOLD, SOCRATIC, DELAY)

Training uses offline Q-learning on observed dialogue transitions.

IMPORTANT LIMITATION:
    In offline/historical data, the action taken was the tutor's actual response,
    which may not match what our policy would have selected. The reward observed
    (extraction change) is associated with the action that was ACTUALLY taken,
    not a hypothetical action. We use the observed action for Q-learning updates
    and clearly distinguish between observed and hypothetical actions.
"""

import json
import logging
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

# Action definitions
ACTION_NAMES = {0: "REDIRECT", 1: "SCAFFOLD", 2: "SOCRATIC", 3: "DELAY"}
NUM_ACTIONS = 4
STATE_DIM = 771


class PolicyNetwork(nn.Module):
    """
    Q-value network for the tutoring intervention policy.

    Architecture as specified:
        Linear(771→256) → ReLU → Linear(256→128) → ReLU →
        Linear(128→64) → ReLU → Linear(64→4)

    Outputs raw Q-values (no softmax).
    """

    def __init__(
        self,
        input_dim: int = STATE_DIM,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = NUM_ACTIONS,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"PolicyNetwork: input={input_dim}, hidden={hidden_dims}, "
            f"output={output_dim}, params={total_params:,}"
        )

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for all actions given a state.

        Args:
            state: State tensor of shape (batch_size, 771) or (771,).

        Returns:
            Q-values tensor of shape (batch_size, 4) or (4,).
        """
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for stabilizing Q-learning updates.

    Stores (state, action, reward, next_state, done) transitions.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self.buffer.append((
            state.detach().cpu(),
            action,
            reward,
            next_state.detach().cpu(),
            done,
        ))

    def sample(self, batch_size: int) -> List[Tuple]:
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class RLTrainer:
    """
    Q-learning trainer for the policy network.

    Implements:
    - Epsilon-greedy action selection
    - Q-learning updates with target network for stability
    - Experience replay
    - Training logging and metrics

    OFFLINE RL LIMITATION:
        When training on historical dialogue data, we record the OBSERVED action
        (what the tutor actually did) and the corresponding reward. The policy
        learns from these observations, but we cannot claim the reward would be
        the same for a different action. This is a fundamental limitation of
        offline/batch RL from observational data.
    """

    def __init__(
        self,
        policy_net: Optional[PolicyNetwork] = None,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        batch_size: int = 64,
        buffer_capacity: int = 10000,
        target_update_freq: int = 10,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            policy_net: The Q-value network. Created if not provided.
            learning_rate: Learning rate for the optimizer.
            gamma: Discount factor for future rewards.
            epsilon: Initial exploration rate.
            epsilon_decay: Multiplicative decay per episode.
            epsilon_min: Minimum exploration rate.
            batch_size: Batch size for replay buffer sampling.
            buffer_capacity: Maximum replay buffer size.
            target_update_freq: Episodes between target network updates.
            device: Torch device (auto-detected if not specified).
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Networks
        self.policy_net = (policy_net or PolicyNetwork()).to(self.device)
        self.target_net = PolicyNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is never trained directly

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.criterion = nn.MSELoss()

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Training metrics
        self.episode_rewards: List[float] = []
        self.episode_losses: List[float] = []
        self.episode_count = 0

        logger.info(
            f"RLTrainer initialized: gamma={gamma}, epsilon={epsilon}, "
            f"lr={learning_rate}, device={self.device}"
        )

    def select_action(
        self,
        state: torch.Tensor,
        epsilon: Optional[float] = None,
        greedy: bool = False,
    ) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: State tensor of shape (771,).
            epsilon: Override exploration rate. Uses self.epsilon if None.
            greedy: If True, always select the best action (no exploration).

        Returns:
            Action ID (0-3).
        """
        eps = 0.0 if greedy else (epsilon if epsilon is not None else self.epsilon)

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)

        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = state.to(self.device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=-1).item()
        self.policy_net.train()

        return action

    def get_q_values(self, state: torch.Tensor) -> List[float]:
        """
        Get Q-values for all actions given a state (no exploration).

        Args:
            state: State tensor of shape (771,).

        Returns:
            List of 4 Q-values.
        """
        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = state.to(self.device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            q_values = self.policy_net(state_tensor)
        return q_values.squeeze(0).cpu().tolist()

    def train_step(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> float:
        """
        Perform a single Q-learning update.

        Uses the Bellman equation:
            Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q_target(s', a') - Q(s, a)]

        In practice, we minimize MSE between Q(s,a) and the target:
            target = r + γ * max_a' Q_target(s', a')  (if not done)
            target = r                                  (if done)

        Args:
            state: Current state tensor (771,).
            action: Action taken (0-3).
            reward: Observed reward.
            next_state: Next state tensor (771,).
            done: Whether the episode is terminated.

        Returns:
            Training loss value.
        """
        # Add to replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)

        # Don't train until we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample a batch
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.stack([t[0] for t in batch]).to(self.device)
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32).to(self.device)
        next_states = torch.stack([t[3] for t in batch]).to(self.device)
        dones = torch.tensor([t[4] for t in batch], dtype=torch.bool).to(self.device)

        # Current Q-values for the actions actually taken
        self.policy_net.train()
        current_q = self.policy_net(states)
        current_q_values = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values using target network
        with torch.no_grad():
            next_q = self.target_net(next_states)
            max_next_q = next_q.max(dim=1)[0]
            # Bellman target: r + γ * max Q(s', a') for non-terminal states
            target_q_values = rewards + self.gamma * max_next_q * (~dones).float()

        # Compute loss and update
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def train_episode(
        self,
        transitions: List[Tuple[torch.Tensor, int, float, torch.Tensor, bool]],
    ) -> Dict[str, float]:
        """
        Train on all transitions from a single dialogue episode.

        Args:
            transitions: List of (state, action, reward, next_state, done) tuples.

        Returns:
            Dictionary with episode metrics (total_reward, avg_loss, num_steps).
        """
        total_reward = 0.0
        total_loss = 0.0
        num_steps = 0

        for state, action, reward, next_state, done in transitions:
            loss = self.train_step(state, action, reward, next_state, done)
            total_reward += reward
            total_loss += loss
            num_steps += 1

        self.episode_count += 1

        # Decay epsilon
        self.epsilon = max(
            self.epsilon_min, self.epsilon * self.epsilon_decay
        )

        # Update target network periodically
        if self.episode_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.debug(
                f"Target network updated at episode {self.episode_count}"
            )

        # Record metrics
        avg_loss = total_loss / max(num_steps, 1)
        self.episode_rewards.append(total_reward)
        self.episode_losses.append(avg_loss)

        return {
            "total_reward": total_reward,
            "avg_loss": avg_loss,
            "num_steps": num_steps,
            "epsilon": self.epsilon,
            "episode": self.episode_count,
        }

    def save(self, path: str) -> None:
        """
        Save the policy network, target network, optimizer state, and training metrics.

        Args:
            path: Path to save the checkpoint (.pth file).
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "episode_rewards": self.episode_rewards,
            "episode_losses": self.episode_losses,
            "gamma": self.gamma,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
        }
        torch.save(checkpoint, str(save_path))
        logger.info(f"Policy saved to {save_path} (episode {self.episode_count})")

    def load(self, path: str) -> None:
        """
        Load a saved policy checkpoint.

        Args:
            path: Path to the checkpoint file.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(str(load_path), map_location=self.device, weights_only=False)

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", 0.05)
        self.episode_count = checkpoint.get("episode_count", 0)
        self.episode_rewards = checkpoint.get("episode_rewards", [])
        self.episode_losses = checkpoint.get("episode_losses", [])
        self.gamma = checkpoint.get("gamma", 0.99)

        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        logger.info(
            f"Policy loaded from {load_path} "
            f"(episode {self.episode_count}, epsilon={self.epsilon:.4f})"
        )

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get a summary of training metrics."""
        if not self.episode_rewards:
            return {"status": "no training data"}

        rewards = np.array(self.episode_rewards)
        return {
            "total_episodes": self.episode_count,
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "avg_reward_last_50": float(np.mean(rewards[-50:])) if len(rewards) >= 50 else float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "current_epsilon": self.epsilon,
        }
