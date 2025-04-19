import sys
import time

import gym_super_mario_bros
import torch
import torch.nn as nn
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import torch.nn.functional as F

from wrappers import *


# Same as duel_dqn.mlp (you can make model.py to avoid duplication.)
class model(nn.Module):
    def __init__(self, n_frame, n_action, device):
        super(model, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 4, 2)
        self.layer3 = nn.Conv2d(64, 64, 3, 1)
        
        self.adv_fc1 = nn.Linear(3136, 512)
        self.adv_fc2 = nn.Linear(512, n_action)
        
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


def arrange(s):
    if not type(s) == "numpy.ndarray":
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    return np.expand_dims(ret, 0)


if __name__ == "__main__":
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/mario_q_target.pth"
    print(f"Load ckpt from {ckpt_path}")
    n_frame = 4
    
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = model(n_frame, env.action_space.n, device).to(device)

    q.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    total_score = 0.0
    done = False
    s = arrange(env.reset())
    i = 0
    while not done:
        env.render()
        if device == "cpu":
            a = np.argmax(q(s).detach().numpy())
        else:
            a = np.argmax(q(s).cpu().detach().numpy())
        s_prime, r, done, _ = env.step(a)
        s_prime = arrange(s_prime)
        total_score += r
        s = s_prime
        time.sleep(0.05)

    stage = env.unwrapped._stage
    print("Total score : %f | stage : %d" % (total_score, stage))
