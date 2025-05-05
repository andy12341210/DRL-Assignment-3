import os
import numpy as np
from collections import deque
import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNet(torch.nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc_adv = nn.Linear(64*7*7, 512)
        self.fc_val = nn.Linear(64*7*7, 512)
        self.adv = nn.Linear(512, n_actions)
        self.val = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        adv = F.relu(self.fc_adv(x))
        val = F.relu(self.fc_val(x))
        adv = self.adv(adv)
        val = self.val(val).expand(x.size(0), self.adv.out_features)
        return val + adv - adv.mean(1, keepdim=True)

class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return [self.buffer[i] for i in indices], indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
            self.max_priority = max(self.max_priority, prio)

    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=1e-3,
                 buffer_capacity=10000, batch_size=64, tau=1e-3, steps=3):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.n_step = steps
        self.n_step_buffer = deque(maxlen=steps)
        self.batch_size = batch_size
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.policy_net = QNet(in_channels=4, n_actions=action_size).to(self.device)
        self.target_net = QNet(in_channels=4, n_actions=action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = torch.nn.SmoothL1Loss(reduction="none")

        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.update(hard_update=True)

    def get_action(self, state, epsilon):
      
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if np.random.rand() < epsilon:
          return np.random.choice(self.action_size)

        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def update(self, hard_update=False):
        if hard_update:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def add_experience(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step or done:
            total_reward = 0
            gamma_step = 1.0
            final_next_state = next_state
            final_done = done
            for i in range(len(self.n_step_buffer)):
                total_reward += gamma_step * self.n_step_buffer[i][2]
                gamma_step *= self.gamma
                if self.n_step_buffer[i][4]:
                    final_next_state = self.n_step_buffer[i][3]
                    final_done = True
                    break
            state, action, _, _, _ = self.n_step_buffer[0]
            self.replay_buffer.add((state, action, total_reward, final_next_state, final_done))
            if final_done:
                self.n_step_buffer.clear()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        samples, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            max_next_q = self.target_net(next_states).gather(1, next_actions)
            q_targets = rewards + (1 - dones) * self.gamma * max_next_q

        td_errors = (q_targets - q_values).detach().abs().cpu().numpy().flatten()

        loss = self.criterion(q_values, q_targets)
        loss = (loss * weights).mean()

        self.replay_buffer.update_priorities(indices, td_errors)


        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()


