import gym
import itertools
import numpy as np
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from random import random
from collections import deque, namedtuple
from gan import GAN1d, train_gan


class Trajectory:
    def __init__(self, gamma: float):
        self.gamma = gamma
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.obs_p = []
        self.rewards = []
        self.done = []
        self.q_vals = []
        self.cum_rewards = []
        self.length = 0

    def add(
        self,
        obs: list,
        action: int,
        obs_p: list,
        reward: float,
        done: bool,
        q_val: float,
    ):
        self.obs.append(obs)
        self.actions.append(action)
        self.obs_p.append(obs_p)
        self.rewards.append(reward)
        self.done.append(done)
        self.q_vals.append(q_val)
        # Update cumulative awards. This may be prohobatively slow for long episodes, we will see!
        self.cum_rewards = [
            self.cum_rewards[i] + reward * (self.gamma ** (self.length - i))
            for i in range(self.length)
        ]
        self.cum_rewards.append(reward)
        self.length += 1

    def mse(self):
        np_q_vals = np.array(self.q_vals)
        np_cum_rewards = np.array(self.cum_rewards)
        return np.sum(np.square([np_q_vals, np_cum_rewards])) / self.length


class ReplayBuffer:
    def __init__(self, buffer_len: int):
        self.obs_buffer = deque([], buffer_len)
        self.obs_prime_buffer = deque([], buffer_len)
        self.reward_buffer = deque([], buffer_len)
        self.actions_buffer = deque([], buffer_len)
        self.done_buffer = deque([], buffer_len)
        self.buffer_len = buffer_len
        self.length = 0

    def add(self, obs: list, action: int, obs_prime: list, reward: float, done: bool):
        self.obs_buffer.append(obs)
        self.actions_buffer.append(action)
        self.obs_prime_buffer.append(obs_prime)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        if self.length < self.buffer_len:
            self.length += 1

    def add_trajectory(self, trajectory: Trajectory):
        self.obs_buffer.extend(trajectory.obs)
        self.actions_buffer.extend(trajectory.actions)
        self.obs_prime_buffer.extend(trajectory.obs_p)
        self.reward_buffer.extend(trajectory.rewards)
        self.done_buffer.extend(trajectory.done)
        self.length += trajectory.length
        if self.length > self.buffer_len:
            self.length = self.buffer_len

    def sample(self, batch_size : int):
        perm = np.random.permutation(self.length)[:batch_size]
        return (
            torch.FloatTensor(self.obs_buffer)[perm],
            torch.LongTensor(self.actions_buffer)[perm],
            torch.FloatTensor(self.obs_prime_buffer)[perm],
            torch.FloatTensor(self.reward_buffer)[perm],
            torch.FloatTensor(self.done_buffer)[perm],
        )



class SimpleEstimator(nn.Module):
    def __init__(self, action_dim: int, obs_size: int, hidden_size: int):
        super(SimpleEstimator, self).__init__()
        self.action_dim = action_dim
        self.obs_size = obs_size
        self.input_size = obs_size
        self.hidden_size = hidden_size
        self.predict = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_dim),
        )

    def forward(self, obs: torch.FloatTensor):
        return self.predict(obs)


def train_ddqn(
    env: gym.Env,
    valid_actions: list,
    q_est: nn.Module,
    q_target: nn.Module,
    num_frames: int,
    batch_size: int = 4096,
    gamma: float = 0.9,
    buffer_size: int = 100000,
) -> list:
    """
    Train q_est with on gym env for num_frames iterations
    TODO: get rid of q_target argument
    TODO: add optim as arg
    """

    rb = ReplayBuffer(buffer_size)
    epsilon = 1.0
    observation = env.reset()
    criterion = nn.MSELoss()
    losses = []
    optimizer = optim.Adam(q_est.parameters(), lr=0.0001)
    min_loss = 100.0
    for frame in range(num_frames):
        # Sample from environment!
        torchy_obs = torch.FloatTensor(observation).unsqueeze(0)
        if random() < epsilon:
            action = env.action_space.sample()
        else:
            pred = q_est(torchy_obs)
            action = int(pred.argmax())
        observation_prime, reward, done, info = env.step(action)
        rb.add(observation, action, observation_prime, reward, (1.0 if done else 0.0))
        if done:
            observation = env.reset()
        else:
            observation = observation_prime
        if epsilon > 0.05:
            epsilon -= 0.95 / 100000

        # Train the beast
        if (frame + 1) % batch_size == 0:
            q_est.zero_grad()

            # Sample the things you need from the buffer!
            obs, actions, obs_p, rews, done = rb.sample(batch_size)

            # Choose the argmax of actions from obs_prime
            q_vals_mat = q_est(obs_p, valid_actions[0])
            max_actions = valid_actions[0]
            for action in valid_actions[1:]:
                q_vals_mat = torch.cat((q_vals_mat, q_est(obs_p, action)), dim=1)
            max_actions = torch.argmax(q_vals_mat, 1)

            # Find MSE between target and network
            target = rews.unsqueeze(1) + (1 - done.unsqueeze(1)) * gamma * q_target(
                obs_p, max_actions
            )
            prediction = q_est(obs, actions)
            loss = criterion(target, prediction)
            losses.append(float(loss))
            if float(loss) < min_loss:
                torch.save(q_est, "best_so_far.pt")
            loss.backward()
            optimizer.step()

        # Update target every 10000 steps, can do moving avg later
        if (frame + 1) % 25000 == 0:
            print("Copying over at iter: ", frame, "with loss: ", loss)
            q_target = copy.deepcopy(q_est)
            for p in q_target.parameters():
                p.requires_grad = False
    env.close()
    return losses


def train_ddqn_and_gan(
    env: gym.Env,
    valid_actions: list,
    q_est: nn.Module,
    q_target: nn.Module,
    gan: nn.Module,
    optimizer: optim,
    num_frames: int,
    batch_size: int = 4096,
    gamma: float = 0.9,
    buffer_size: int = 100000,
):
    # TODO: Fix typing in file
    # TODO: Note: this probably isn't the right place for this function
    rb = ReplayBuffer(buffer_size)
    epsilon = 1.0
    observation = env.reset()
    criterion = nn.MSELoss()
    ddqn_losses = []
    generator_losses = []
    discriminator_losses = []
    min_loss = 100.0
    id_ses = []
    ood_ses = []
    # Initialize trajectory tracking
    trajectory = Trajectory(gamma)
    frames_since_accuracy_check = 0
    for frame in range(num_frames):
        frames_since_accuracy_check += 1
        # Sample from environment!
        torchy_obs = torch.FloatTensor(observation).unsqueeze(0)
        if random() < epsilon:
            action = env.action_space.sample()
            max_val = q_est(torchy_obs)[0][action]
        else:
            pred = q_est(torchy_obs)
            max_val = pred.max()
            action = int(pred.argmax())
        observation_prime, reward, done, info = env.step(action)
        trajectory.add(
            observation,
            action,
            observation_prime,
            reward,
            (1.0 if done else 0.0),
            max_val,
        )

        if done:
            rb.add_trajectory(trajectory)
            # Measure/record accuracy stuff here
            if frames_since_accuracy_check > 500:
                frames_since_accuracy_check = 0
                q_vals_t = torch.FloatTensor(trajectory.q_vals)
                cum_rewards_t = torch.FloatTensor(trajectory.cum_rewards)
                rewards_t = torch.FloatTensor(trajectory.rewards)
                se = (q_vals_t - cum_rewards_t).pow(2)
                labels = gan.discriminator(torch.FloatTensor(trajectory.obs))
                id_weights = labels[:, 0]
                ood_weights = labels[:, 1]
                ood_se = torch.sum(se * ood_weights)
                id_se = torch.sum(se * id_weights)
                ood_ses.append(float(ood_se / torch.sum(ood_weights)))
                id_ses.append(float(id_se / torch.sum(id_weights)))

            trajectory.clear()
            observation = env.reset()
        else:
            observation = observation_prime
        if epsilon > 0.05:
            epsilon -= 0.95 / 100000

        # Training time
        if (frame + 1) % batch_size == 0 and frame > 2 * batch_size:
            q_est.zero_grad()
            # Sample the things you need from the buffer!
            obs, actions, obs_p, rews, done = rb.sample(batch_size)

            # TRAIN THE GAN <('_'<)
            generator_loss, discriminator_loss = train_gan(
                gan, [obs], 1, train_noise=0.05
            )
            generator_losses.extend(generator_loss)
            discriminator_losses.extend(discriminator_loss)

            # Choose the argmax of actions from obs_prime
            q_vals_mat = q_est(obs_p)
            max_actions = torch.argmax(q_vals_mat, 1)
            # Find MSE between target and network
            q_target_pred = q_target(obs_p).gather(1, max_actions.unsqueeze(1))
            target = rews.unsqueeze(1) + (1 - done.unsqueeze(1)) * gamma * q_target_pred
            prediction = q_est(obs)[actions]
            loss = criterion(target, prediction)
            ddqn_losses.append(float(loss))
            if float(loss) < min_loss:
                torch.save(q_est, "best_so_far.pt")
            loss.backward()
            optimizer.step()

        # Update target every 25000 steps, can do moving avg later
        if (frame + 1) % 25000 == 0:
            print("Copying over at iter: ", frame, "with loss: ", loss)
            q_target = copy.deepcopy(q_est)
            for p in q_target.parameters():
                p.requires_grad = False
    env.close()
    return (
        ddqn_losses,
        generator_losses,
        discriminator_losses,
        id_ses,
        ood_ses,
    )


def test_ddqn_network(
    env: gym.Env, valid_actions: list, q_est: nn.Module, num_frames: int = 100
):
    observation = env.reset()
    num_resets = 0
    cum_reward = 0
    for frame in range(num_frames):
        # Sample from environment!
        env.render()
        torchy_obs = torch.FloatTensor(observation).unsqueeze(0)
        max_val = q_est(torchy_obs, valid_actions[0])[0]
        max_action = valid_actions[0]
        for action in valid_actions[1:]:
            q_val = q_est(torchy_obs, action)[0]
            if q_val > max_val:
                max_action = action
                max_val = q_val
        action = max_action
        observation, reward, done, info = env.step(action)
        cum_reward += reward
        if done:
            print("Reset with reward: ", reward)
            print("and cumulative reward: ", cum_reward)
            cum_reward = 0
            num_resets += 1
            observation = env.reset()
    env.close()
    return num_resets
