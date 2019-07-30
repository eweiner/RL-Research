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
import time
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
        self.obs.append(copy.deepcopy(obs))
        self.actions.append(action)
        self.obs_p.append(copy.deepcopy(obs_p))
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

    def sample(self, batch_size: int):
        perm = torch.LongTensor(np.random.choice(self.length, batch_size))
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
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_dim),
        )

    def forward(self, obs: torch.FloatTensor):
        obs = obs.view(obs.shape[0], -1)
        return self.predict(obs)


class Conv1dEstimator(nn.Module):
    def __init__(
        self,
        action_dim: int,
        obs_size: int,
        hidden_size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ):
        super(Conv1dEstimator, self).__init__()
        self.action_dim = action_dim
        self.obs_size = obs_size
        self.input_size = obs_size
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.max_pool_kernel = 2
        self.conv_output = self.obs_size
        self.conv_out = 64
        self.conv_predict = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size),
            nn.MaxPool1d(self.max_pool_kernel),
            nn.ReLU(),
            nn.Conv1d(self.out_channels, self.out_channels * 2, self.kernel_size),
            nn.MaxPool2d(self.max_pool_kernel),
            nn.ReLU(),
        )
        self.linear_predict = nn.Sequential(
            nn.Linear(self.conv_out, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_dim),
        )

    def forward(self, obs: torch.FloatTensor):
        x = self.conv_predict(obs)
        x = x.view(x.size(0), -1)
        return self.linear_predict(x)


def flatten_deque(deque_to_flatten: deque) -> list:
    flattened = []
    for e in deque_to_flatten:
        flattened.extend(e)
    return flattened


def train_ddqn_and_gan(
    env: gym.Env,
    q_est: nn.Module,
    q_target: nn.Module,
    gan: nn.Module,
    optimizer: optim,
    num_frames: int,
    consecutive_observations=4,
    batch_size: int = 4096,
    gamma: float = 0.9,
    buffer_size: int = 100000,
    epsilon_start: float = 1,
    rewards_tracking: list = [],
    ddqn_losses: list = [],
    generator_losses: list = [],
    discriminator_losses: list = [],
):
    # TODO: Fix typing in file
    # TODO: Note: this probably isn't the right place for this function
    rb = ReplayBuffer(buffer_size)
    # alpha = 0.999 # For slow changing target network
    epsilon = epsilon_start
    observation = env.reset()
    criterion = nn.MSELoss()
    reward_tracking = 0
    max_reward_so_far = 0
    id_ses = []
    ood_ses = []
    loss = None
    num_trajectories = 0
    obs_list = deque(
        [observation for i in range(consecutive_observations)], consecutive_observations
    )
    obs_p_list = deque(
        [observation for i in range(consecutive_observations)], consecutive_observations
    )
    freeze_generator = False
    # Initialize trajectory tracking
    trajectory = Trajectory(gamma)
    frames_since_accuracy_check = 0
    for frame in range(num_frames):
        # for t_param, e_param in zip(q_target.parameters(), q_est.parameters()):
        #     t_param.data = alpha * t_param.data + (1 - alpha) * e_param.data
        frames_since_accuracy_check += 1
        # Sample from environment!
        obs_list.append(observation)
        torchy_obs = torch.FloatTensor(obs_list).unsqueeze(0)
        if random() < epsilon:
            action = env.action_space.sample()
            max_val = q_est(torchy_obs)[0][action]
        else:
            pred = q_est(torchy_obs)
            max_val = pred.max()
            action = int(torch.argmax(pred[0]))
        observation_prime, reward, done, info = env.step(action)
        reward_tracking += reward
        obs_p_list.append(observation_prime)
        trajectory.add(
            obs_list, action, obs_p_list, reward, (1.0 if done else 0.0), max_val
        )
        # rb.add(
        #     obs_list,
        #     action,
        #     obs_p_list,
        #     reward,
        #     (1.0 if done else 0.0),
        # )

        if done:
            num_trajectories += 1
            rb.add_trajectory(trajectory)
            rewards_tracking.append(reward_tracking)
            reward_tracking = 0
            # Measure/record accuracy stuff here
            # if frames_since_accuracy_check > 500:
            #     frames_since_accuracy_check = 0
            #     q_vals_t = torch.FloatTensor(trajectory.q_vals)
            #     cum_rewards_t = torch.FloatTensor(trajectory.cum_rewards)
            #     rewards_t = torch.FloatTensor(trajectory.rewards)
            #     se = (q_vals_t - cum_rewards_t).pow(2)
            #     labels = gan.discriminator(torch.FloatTensor(trajectory.obs))
            #     id_weights = labels[:, 0]
            #     ood_weights = labels[:, 1]
            #     ood_se = torch.sum(se * ood_weights)
            #     id_se = torch.sum(se * id_weights)
            #     ood_ses.append(float(ood_se / torch.sum(ood_weights)))
            #     id_ses.append(float(id_se / torch.sum(id_weights)))

            trajectory.clear()
            observation = env.reset()
        else:
            observation = observation_prime
        if epsilon > 0.05:
            epsilon -= 0.95 / 100000

        # Training time
        if (frame + 1) % batch_size == 0 and frame > buffer_size // 2 and rb.length > 0:
            # start = time.time()
            q_est.zero_grad()
            if rewards_tracking[-1] > max_reward_so_far:
                max_reward_so_far = rewards_tracking[-1]
                torch.save(q_est, "best_model_yet.pt")
            # Sample the things you need from the buffer!
            obs, actions, obs_p, rews, done = rb.sample(batch_size)
            # TRAIN THE GAN <('_'<)
            generator_loss, discriminator_loss = train_gan(
                gan,
                obs.unsqueeze(0),
                1,
                train_noise=0.05,
                freeze_generator=freeze_generator,
            )
            generator_losses.extend(generator_loss)
            discriminator_losses.extend(discriminator_loss)
            freeze_generator = not freeze_generator

            # DQN CODE
            # Choose the argmax of actions from obs_prime
            # q_est_vals = q_est(obs)
            # prediction = q_est(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
            # q_target_pred, _ = torch.max(q_target(obs_p), 1)
            # target = rews + (1 - done) * gamma * q_target_pred
            # loss = criterion(target, prediction)
            # DDQN CODE
            q_vals_mat = q_est(obs_p)
            max_actions = torch.argmax(q_vals_mat, 1)
            # Find MSE between target and network
            q_target_pred = q_target(obs_p).gather(1, max_actions.unsqueeze(1))
            target = rews.unsqueeze(1) + (1 - done.unsqueeze(1)) * gamma * q_target_pred
            prediction = q_est(obs).gather(1, actions.unsqueeze(1))
            loss = criterion(target, prediction)
            ddqn_losses.append(float(loss))
            loss.backward()
            optimizer.step()
        # Update target every 25000 steps, can do moving avg later
        if (frame + 1) % 10000 == 0 and loss is not None:
            print("Copying over at iter: ", frame, "with loss: ", loss)
            print("Running a test trajectory...")
            test_done = False
            test_observation = env.reset()
            test_obs_list = deque(
                [test_observation for i in range(consecutive_observations)],
                consecutive_observations,
            )
            running_reward = 0
            while not test_done:
                test_obs_list.append(test_observation)
                torchy_obs = torch.FloatTensor(test_obs_list).unsqueeze(0)
                pred = q_est(torchy_obs)
                max_val = pred[0].max()
                action = torch.argmax(pred[0]).numpy()
                test_observation, reward, test_done, info = env.step(action)
                running_reward += reward
                if test_done:
                    observation = env.reset()
                    print("Test Trajectory had cumulative reward: ", running_reward)

            q_target = copy.deepcopy(q_est)
            for p in q_target.parameters():
                p.requires_grad = False
    env.close()
    return (
        rewards_tracking,
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
