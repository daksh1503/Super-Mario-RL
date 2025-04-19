import pickle
import random
from collections import deque

import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import time

from wrappers import *


def arrange(s):
    if not type(s) == "numpy.ndarray":
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    return np.expand_dims(ret, 0)


class replay_memory(object):
    def __init__(self, N):
        self.memory = deque(maxlen=N)
        self.priorities = deque(maxlen=N)
        self.alpha = 0.8  # Priority exponent
        self.beta = 0.4   # Importance sampling weight
        self.beta_increment = 0.002
        self.epsilon = 1e-6  # Small constant to prevent zero probabilities

    def push(self, transition, priority=None):
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        self.memory.append(transition)
        self.priorities.append(priority)

    def sample(self, n):
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), n, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights

    def __len__(self):
        return len(self.memory)


class model(nn.Module):
    def __init__(self, n_frame, n_action, device):
        super(model, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 4, 2)
        self.layer3 = nn.Conv2d(64, 64, 3, 1)
        
        # Advantage stream
        self.adv_fc1 = nn.Linear(3136, 512)
        self.adv_fc2 = nn.Linear(512, n_action)
        
        # Value stream
        self.val_fc1 = nn.Linear(3136, 512)
        self.val_fc2 = nn.Linear(512, 1)
        
        self.device = device
        self.apply(init_weights)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
            
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(-1, 3136)

        adv = F.relu(self.adv_fc1(x))
        adv = self.adv_fc2(adv)

        val = F.relu(self.val_fc1(x))
        val = self.val_fc2(val)

        return val + adv - adv.mean(1, keepdim=True)


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train(q, q_target, memory, batch_size, gamma, optimizer, device):
    samples, indices, weights = memory.sample(batch_size)
    s, r, a, s_prime, done = list(map(list, zip(*samples)))
    weights = torch.FloatTensor(weights).to(device)
    
    s = np.array(s).squeeze()
    s_prime = np.array(s_prime).squeeze()
    
    # Double DQN: Use online network to select action, target network to evaluate
    with torch.no_grad():
        # Select actions using online network
        online_next_q = q(s_prime)
        next_actions = online_next_q.max(1)[1]
        
        # Evaluate actions using target network
        next_q_values = q_target(s_prime).gather(1, next_actions.unsqueeze(1))
        
        # Calculate target Q values
        r = torch.FloatTensor(r).unsqueeze(-1).to(device)
        done = torch.FloatTensor(done).unsqueeze(-1).to(device)
        y = r + gamma * next_q_values * done

    # Calculate current Q values and loss
    a = torch.tensor(a).unsqueeze(-1).to(device)
    q_values = q(s).gather(1, a.view(-1, 1).long())
    
    # Calculate TD errors for prioritized replay
    td_errors = torch.abs(y - q_values).detach().cpu().numpy()
    
    # Update priorities
    for idx, error in zip(indices, td_errors):
        memory.priorities[idx] = error[0] + memory.epsilon
    
    # Calculate weighted loss
    loss = (weights * F.smooth_l1_loss(q_values, y, reduction='none', beta=0.2)).mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q.parameters(), 5)  # Gradient clipping
    optimizer.step()
    
    return loss


def copy_weights(q, q_target):
    q_dict = q.state_dict()
    q_target.load_state_dict(q_dict)


def main(env, q, q_target, optimizer, device):
    t = 0
    gamma = 0.995
    batch_size = 256  # Reduced batch size for more frequent updates
    N = 200000  # Increased replay buffer
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 0.9999
    eps = eps_start
    
    memory = replay_memory(N)
    update_interval = 2000  # More frequent target updates
    print_interval = 10

    score_lst = []
    total_score = 0.0
    loss = 0.0
    start_time = time.perf_counter()

    for k in range(1000000):
        s = arrange(env.reset())
        done = False

        while not done:
            if eps > np.random.rand():
                a = env.action_space.sample()
            else:
                if device == "cpu":
                    a = np.argmax(q(s).detach().numpy())
                else:
                    a = np.argmax(q(s).cpu().detach().numpy())
            env.render()
            s_prime, r, done, info = env.step(a)
            s_prime = arrange(s_prime)
            total_score += r
            r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) * 1.5
            if done:
                r -= 30  # Penalty for dying
            if info.get('flag_get', False):
                r += 150  # Bonus for completing level
            if info.get('x_pos', 0) > info.get('max_x', 0):  # If new max position reached
                r += 0.5  # Small bonus for progress    
            memory.push((s, float(r), int(a), s_prime, int(1 - done)))
            s = s_prime
            stage = env.unwrapped._stage
            if len(memory) > 1000:
                loss += train(q, q_target, memory, batch_size, gamma, optimizer, device)
                t += 1
            if t % update_interval == 0:
                copy_weights(q, q_target)
                torch.save(q.state_dict(), "mario_q.pth")
                torch.save(q_target.state_dict(), "checkpoints/mario_q_target.pth")


        if k % print_interval == 0:
            time_spent, start_time = time.perf_counter() - start_time, time.perf_counter()
            print(
                "%s |Epoch : %d | score : %f | loss : %.2f | stage : %d | time spent: %f"
                % (
                    device,
                    k,
                    total_score / print_interval,
                    loss / print_interval,
                    stage,
                    time_spent,
                )
            )
            score_lst.append(total_score / print_interval)
            total_score = 0
            loss = 0.0
            pickle.dump(score_lst, open("score.p", "wb"))

        # Update epsilon decay in the training loop
        eps = max(eps_end, eps * eps_decay)


if __name__ == "__main__":
    n_frame = 4
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    q = model(n_frame, env.action_space.n, device).to(device)
    q_target = model(n_frame, env.action_space.n, device).to(device)
    optimizer = optim.Adam(q.parameters(), lr=0.0001,  eps=1.5e-4)
    print(device)

    main(env, q, q_target, optimizer, device)


