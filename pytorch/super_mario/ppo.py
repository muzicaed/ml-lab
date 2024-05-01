import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical

DEFAULT_SAVE_DIR = '/Users/mikaelhellman/dev/ml-lab/model/ppo'


class ActorNetwork(nn.Module):
    def _init__(self, no_of_actions, input_dims, learning_rate, fc1=256, fc2=256, save_dir=DEFAULT_SAVE_DIR):
        super(ActorNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Linear(fc2, no_of_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), learning_rate)
        self.save_file = os.path.join(save_dir, 'actor_ppo')

    def forward(self, state):
        disttribution = self.actor(state)
        return Categorical(disttribution)

    def save(self):
        T.save(self.state_dict(), self.save_file)

    def load(self):
        self.load_state_dict(T.load(self.save_file))


class CriticNetwork(nn.Module):
    def _init__(self, input_dims, learning_rate, fc1=256, fc2=256, save_dir=DEFAULT_SAVE_DIR):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Linear(fc2, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), learning_rate)
        self.save_file = os.path.join(save_dir, 'critic_ppo')

    def forward(self, state):
        return self.critic(state)

    def save(self):
        T.save(self.state_dict(), self.save_file)

    def load(self):
        self.load_state_dict(T.load(self.save_file))


class PPOMemory:

    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def create_batches(self):
        states_count = len(self.states)
        batch_start = np.arange(0, states_count, self.batch_size)
        indices = np.arange(states_count, dtype=np. int64)
        np.random.shuffle(indices)
        batches = [indices[i: i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), \
            np.array(self.probs), np.array(self.values), \
            np.array(self.rewards), np.array(self.dones), batches

    def store(self, state, action, probs, values, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []
