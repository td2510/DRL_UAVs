"""
DDPG (Deep Deterministic Policy Gradient) agent implementation in PyTorch.

Architecture follows the reference mec_drl implementation but modernized:
- Actor: state -> action (sigmoid output scaled to bounds)
- Critic: (state, action) -> Q-value
- Target networks with soft update
- Replay buffer and OU noise for exploration
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import config as cfg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAYER1 = 400
LAYER2 = 300


class ActorNetwork(nn.Module):
    """
    Maps state to action.
    Output uses tanh activation to keep actions in [-1, 1].
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, LAYER1),
            nn.LayerNorm(LAYER1),
            nn.ReLU(),
            nn.Linear(LAYER1, LAYER2),
            nn.LayerNorm(LAYER2),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(LAYER2, action_dim)

        # Initialize output weights to small values (matches mec_drl)
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_layer.bias, -3e-3, 3e-3)

    def forward(self, state):
        x = self.net(state)
        # tanh keeps output in [-1, 1]
        return torch.tanh(self.output_layer(x))


class CriticNetwork(nn.Module):
    """
    Maps (state, action) to Q-value.
    Action is injected at the second hidden layer (matches mec_drl structure).
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Process state
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, LAYER1),
            nn.LayerNorm(LAYER1),
            nn.ReLU(),
        )

        # Combine state features with action
        self.state_fc = nn.Linear(LAYER1, LAYER2)
        self.action_fc = nn.Linear(action_dim, LAYER2)

        self.combined_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(LAYER2, 1),
        )

        # Initialize output weights
        nn.init.uniform_(self.combined_net[1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.combined_net[1].bias, -3e-3, 3e-3)

    def forward(self, state, action):
        s = self.state_net(state)
        # Inject action at second layer (like mec_drl)
        x = self.state_fc(s) + self.action_fc(action)
        return self.combined_net(x)


class ReplayBuffer:
    """Experience replay buffer (same logic as mec_drl)."""

    def __init__(self, capacity, seed=123):
        self.buffer = deque(maxlen=capacity)
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
        )

    def size(self):
        return len(self.buffer)


class OrnsteinUhlenbeckNoise:
    """
    OU noise for exploration.
    Based on: http://math.stackexchange.com/questions/1287634
    """

    def __init__(self, size, mu=0.0, sigma=0.2, theta=0.15, dt=1e-2):
        self.mu = np.full(size, mu)
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.size = size
        self.reset()

    def reset(self):
        self.x_prev = self.mu.copy()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        )
        self.x_prev = x
        return x


class DDPGAgent:
    """
    DDPG agent for UAV trajectory optimization.
    """

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor networks
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.actor_target = ActorNetwork(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.ACTOR_LR)

        # Critic networks
        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.CRITIC_LR)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(cfg.BUFFER_SIZE)

        # Exploration noise
        self.noise = OrnsteinUhlenbeckNoise(
            size=action_dim,
            sigma=cfg.NOISE_SIGMA,
            theta=cfg.NOISE_THETA,
        )

        self.batch_size = cfg.BATCH_SIZE
        self.gamma = cfg.GAMMA
        self.tau = cfg.TAU_SOFT

    def predict(self, state, add_noise=True):
        """Select action given state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()

        if add_noise:
            action += self.noise()

        return np.clip(action, -1.0, 1.0)

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        """
        Sample a batch and update actor/critic networks.
        Returns critic loss for logging.
        """
        if self.replay_buffer.size() < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.FloatTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        dones_t = torch.FloatTensor(dones).to(device)

        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q = self.critic_target(next_states_t, next_actions)
            y = rewards_t + (1 - dones_t) * self.gamma * target_q

        current_q = self.critic(states_t, actions_t)
        critic_loss = nn.MSELoss()(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update Actor
        predicted_actions = self.actor(states_t)
        actor_loss = -self.critic(states_t, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return critic_loss.item()

    def _soft_update(self, source, target):
        """Polyak averaging for target networks."""
        for target_param, source_param in zip(
            target.parameters(), source.parameters()
        ):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path):
        """Save model checkpoints."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        """Load model checkpoints."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def reset_noise(self):
        """Reset OU noise at the start of each episode."""
        self.noise.reset()
