# Adapted from https://github.com/rlcode/per

import random
import torch
import numpy as np
from SumTree import SumTree
from torch.autograd import Variable

class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        priorities = np.array(list(map(lambda x: self.e if x == 0 else x, priorities)))
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


    def add_trajectory(self, model_est, model_target, trajectory):
        for i in range(trajectory.length):
            obs = trajectory.obs[i]
            action = trajectory.actions[i]
            obs_prime = trajectory.obs_p[i]
            reward = trajectory.rewards[i]
            done = trajectory.done[i]

            target = model_est(Variable(torch.FloatTensor(obs))).data

            old_val = target[action]
            target_val = model_target(Variable(torch.FloatTensor(obs_prime))).data
            if done:
                target[action] = reward
            else:
                target[action] = reward + trajectory.gamma * torch.max(target_val)

            error = abs(old_val - target[action])
            self.add(error, (obs, action, reward, obs_prime, done))


