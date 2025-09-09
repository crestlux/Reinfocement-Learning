import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import argparse
from collections import deque
import random, math

plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
rewards_history = []
avg_rewards_history = []
episodes_axis = []
RUN_LABEL = ""

def make_run_label(args) -> str:
    rb = "on" if args.replay_buffer else "off"
    if args.target_network:
        if args.target_update_type == "soft":
            tgt = f"soft, τ={args.tau:g}"
        else:
            tgt = f"hard, f={args.hard_update_freq}"
    else:
        tgt = "off"
    return f"replay_buffer={rb} | target_network_update={tgt}"

def plot_rewards(episode: int, reward: float, avg_reward: float):
    rewards_history.append(reward)
    avg_rewards_history.append(avg_reward)
    episodes_axis.append(episode)
    ax.clear()
    ax.plot(episodes_axis, rewards_history, label='Episode Reward', alpha=0.6)
    ax.plot(episodes_axis, avg_rewards_history, label='Avg 100-Episode Reward', linewidth=2)
    ax.set_title(f'Training DQN... Episode: {episode}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend(loc='upper left')
    ax.text(0.99, 0.98, RUN_LABEL, transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle="round", alpha=0.2))
    ax.grid(True)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ReplayBuffer:
    def __init__(
        self, state_shape, action_dim, max_size=int(1e6), discrete_action=False, 
        device=None,dtype=np.float32, sample_replace=True
    ):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.discrete = bool(discrete_action)
        self.sample_replace = bool(sample_replace)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None else torch.device(device)
        )
        self.state = np.empty((self.max_size, *state_shape), dtype=dtype)
        self.next_state = np.empty((self.max_size, *state_shape), dtype=dtype)

        if self.discrete:
            self.action = np.empty((self.max_size, 1), dtype=np.int64)
        else:
            self.action = np.empty((self.max_size, action_dim), dtype=dtype)

        self.reward = np.empty((self.max_size, 1), dtype=dtype)
        self.done   = np.empty((self.max_size, 1), dtype=np.float32)
        self._rng = np.random.default_rng()

    def push(self, state, action, reward, next_state, done):
        self.state[self.ptr]      = state
        self.next_state[self.ptr] = next_state
        if self.discrete:
            self.action[self.ptr, 0] = int(action)
        else:
            self.action[self.ptr]    = action
        self.reward[self.ptr, 0] = float(reward)
        self.done[self.ptr, 0]   = float(done)

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if self.sample_replace:
            ind = self._rng.integers(0, self.size, size=batch_size)        # with replacement
        else:
            ind = self._rng.choice(self.size, size=batch_size, replace=False)  # without replacement

        s  = torch.as_tensor(self.state[ind], device=self.device)
        ns = torch.as_tensor(self.next_state[ind], device=self.device)
        r  = torch.as_tensor(self.reward[ind], device=self.device)
        d  = torch.as_tensor(self.done[ind], device=self.device)

        if self.discrete:
            a = torch.as_tensor(self.action[ind], device=self.device, dtype=torch.long)
        else:
            a = torch.as_tensor(self.action[ind], device=self.device)

        return s, a, r, ns, d

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, state_dim, action_dim, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = args.gamma
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay = args.epsilon_decay
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.learning_starts = args.learning_starts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_replay_buffer = bool(args.replay_buffer)
        self.use_target_network = bool(args.target_network)
        
        if self.use_replay_buffer:
            self.memory = ReplayBuffer(state_shape=(state_dim,), action_dim=1, 
                            max_size=args.memory_size, discrete_action=True, device=self.device)
        else:
            self.memory = None
            self.last_transition = None
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        
        if self.use_target_network:
            self.target_model = DQN(state_dim, action_dim).to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
            self.target_update_type = args.target_update_type
            self.tau = args.tau
            if self.target_update_type == "soft":
                assert 0.0 < self.tau <= 1.0
        else:
            self.target_model = self.model
            self.target_update_type = "none"

        self.steps_done = 0

    @torch.no_grad()
    def hard_update(self):
        """Target <- Online (hard copy)"""
        if not self.use_target_network: return
        self.target_model.load_state_dict(self.model.state_dict())

    @torch.no_grad()
    def soft_update(self):
        """Target <- (1-tau)*Target + tau*Online (Polyak/EMA)"""
        if not self.use_target_network: return
        tau = self.tau
        for tp, p in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)

    def select_action(self, state):
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) \
                * math.exp(-1. * self.steps_done / max(1.0, self.epsilon_decay))

        # decide action with epsilon-greedy policy
        if random.random() < eps:
            action = random.randrange(self.action_dim)  # exploration
        else:
            was_training = self.model.training
            try:
                self.model.eval()
                with torch.no_grad():
                    state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device) \
                            if not isinstance(state, torch.Tensor) else state.to(self.device, dtype=torch.float32)
                    if state_t.dim() == 1:
                        state_t = state_t.unsqueeze(0)
                    q_values = self.model(state_t)
                    action = int(q_values.argmax(dim=1).item())  # exploitation
            finally:
                if was_training:
                    self.model.train()

        self.steps_done += 1
        return action
    
    def optimize_model(self):
        if self.use_replay_buffer:
            if len(self.memory) < max(self.batch_size, self.learning_starts):
                return
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        else:
            if self.last_transition is None:
                return
            s, a, r, ns, d = self.last_transition
            states = torch.as_tensor(s, device=self.device, dtype=torch.float32).unsqueeze(0)
            actions = torch.as_tensor([[int(a)]], device=self.device, dtype=torch.long)
            rewards = torch.as_tensor([[float(r)]], device=self.device, dtype=torch.float32)
            next_states = torch.as_tensor(ns, device=self.device, dtype=torch.float32).unsqueeze(0)
            dones = torch.as_tensor([[float(d)]], device=self.device, dtype=torch.float32)

        current_q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(dim=1, keepdim=True)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1.0 - dones)
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.use_target_network and self.target_update_type == "soft":
            self.soft_update()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes, default: 1000")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training, default: 64")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor, default: 0.99")
    parser.add_argument("--epsilon_start", type=float, default=0.9, help="Starting value of epsilon, default: 0.9")
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Final value of epsilon, default: 0.05")
    parser.add_argument("--epsilon_decay", type=int, default=5000, help="Decay rate of epsilon, default: 5000")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate, default: 1e-4")
    parser.add_argument("--learning_starts", type=int, default=100, help="Number of steps before starting training, default: 100")    
    parser.add_argument("--memory_size", type=int, default=10000, help="Replay memory size, default: 10000")
    parser.add_argument("--target_update_type", choices=["hard", "soft"], default="soft",help="How to update target network: 'hard' or 'soft' (Polyak/EMA), default: soft")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update factor for target (0<tau<=1). Used when --target_update_type=soft, default: 0.005")
    parser.add_argument("--hard_update_freq", type=int, default=200, help="Target network update frequency (in steps), Used when --target_update_type=hard, default: 200")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, default: 42")
    parser.add_argument(
        "--replay-buffer", action=argparse.BooleanOptionalAction, default=True, 
        help="Enable or disable experience replay (default: enabled)",
    )
    parser.add_argument(
        "--target-network",action=argparse.BooleanOptionalAction, default=True,
        help="Enable or disable target network (default: enabled)",
    )
    args = parser.parse_args()

    global RUN_LABEL
    RUN_LABEL = make_run_label(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make("CartPole-v1")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    obs, info = env.reset(seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(obs_dim, action_dim, args)

    total_steps = 0
    recent_rewards = deque(maxlen=100)

    for ep in range(1, args.episodes + 1):
        state, info = env.reset(seed=args.seed + ep)
        state = np.asarray(state, dtype=np.float32)
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)

            next_state = np.asarray(obs_next, dtype=np.float32)

            if agent.use_replay_buffer:
                agent.memory.push(state, action, float(reward), next_state, float(terminated))
            else:
                agent.last_transition = (state, action, float(reward), next_state, float(terminated))
            agent.optimize_model()

            total_steps += 1

            if agent.use_target_network and agent.target_update_type == "hard" and total_steps % args.hard_update_freq == 0:
                agent.hard_update()

            state = next_state

        recent_rewards.append(ep_reward)
        avg100 = np.mean(recent_rewards) if len(recent_rewards) > 0 else ep_reward
        print(f"Episode {ep:4d} | Reward {ep_reward:.1f} | Avg100 {avg100:.2f} | Steps {agent.steps_done}")
        if ep % 10 == 0:
            plot_rewards(ep, ep_reward, avg100)
        else:
            plt.pause(0.001)

            
    env.close()
    plt.ioff()
    plt.show()
    

if __name__ == "__main__":
    main()