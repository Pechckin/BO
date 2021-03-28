from collections import namedtuple, deque
from rl.utils.utils import *
import numpy as np
import random
import torch


class Buffer(object):
    def __init__(self, capacity, num_workers, device):
        self.num_workers = num_workers
        self.capacity = capacity
        self.Transaction = namedtuple('Transaction', ('s', 'a', 'ri', 're', 'vi', 've', 'lp'))
        self.discounted_reward = RewardForwardFilter(0.99)
        self.reward_rms = RunningMeanStd()
        self.device = device

        self.reset()

    def reset(self):
        self.s, self.cr = [], []
        self.vi, self.a, self.nvi = [], [], []
        self.ve, self.nve = [], []
        self.r, self.m, self.lp = [], [], []

    @staticmethod
    def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def push(self, *args):
        s, vi, ve, a, nvi, nve, r, m, lp, cr = args
        self.s.append(s)
        self.vi.append(vi), self.ve.append(ve)
        self.a.append(a)
        self.nvi.append(nvi), self.nve.append(nve)
        self.r.append(r)
        self.cr.append(cr)
        self.m.append(m)
        self.lp.append(lp)

    def sample(self, num_batches, batch_size):
        total_reward_per_env = np.array(
            [self.discounted_reward.update(reward_per_step.T.numpy()) for reward_per_step in self.cr])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        self.reward_rms.update_from_moments(mean, std ** 2, count)

        sqrt_reward_var = np.sqrt(self.reward_rms.var)

        self.cr = [el / sqrt_reward_var for el in self.cr]

        returns_i = self.compute_gae(self.nvi[-1], self.cr, self.m, self.vi)
        returns_e = self.compute_gae(self.nve[-1], self.r, self.m, self.ve)

        batches = []
        inds = np.random.permutation(np.arange(self.capacity * self.num_workers))
        splitted_inds = np.split(inds, num_batches)
        for ind in splitted_inds:
            assert len(ind) == batch_size
            batch = self.Transaction(s=torch.cat(self.s)[ind].to(self.device),
                                     a=torch.FloatTensor(self.a).view(-1, 1)[ind].to(self.device),
                                     ri=torch.cat(returns_i)[ind].to(self.device),
                                     re=torch.cat(returns_e)[ind].to(self.device),
                                     vi=torch.cat(self.vi)[ind].to(self.device),
                                     ve=torch.cat(self.ve)[ind].to(self.device),
                                     lp=torch.cat(self.lp)[ind].to(self.device))

            batches.append(batch)

        self.reset()
        return batches

    def __len__(self):
        return len(self.memory)

    def full(self):
        if len(self.s) < self.capacity:
            return False
        return True
