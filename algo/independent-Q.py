import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import random

import itertools
import matplotlib.pyplot as plt
import math
import magent
import pettingzoo
import argparse
import time
import os
import datetime

from gym.spaces import Discrete, Box
from pettingzoo.magent.render import Renderer
from pettingzoo.utils import agent_selector, wrappers
from gym.utils import seeding
from pettingzoo.utils.env import ParallelEnv
from torch.utils.tensorboard import SummaryWriter


def make_env(raw_env):
    def env_fn(**kwargs):
        env = raw_env(**kwargs)
        env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env
    return env_fn


class magent_parallel_env(ParallelEnv):
    def __init__(self, env, active_handles, names, map_size, max_cycles, reward_range, minimap_mode, extra_features):
        self.map_size = map_size
        self.max_cycles = max_cycles
        self.minimap_mode = minimap_mode
        self.extra_features = extra_features
        self.env = env
        self.handles = active_handles
        self._all_handles = self.env.get_handles()
        env.reset()
        self.generate_map()
        self.team_sizes = team_sizes = [env.get_num(handle) for handle in self.handles]
        self.agents = [f"{names[j]}_{i}" for j in range(len(team_sizes)) for i in range(team_sizes[j])]
        self.possible_agents = self.agents[:]

        num_actions = [env.get_action_space(handle)[0] for handle in self.handles]
        action_spaces_list = [Discrete(num_actions[j]) for j in range(len(team_sizes)) for i in range(team_sizes[j])]
        # may change depending on environment config? Not sure.
        team_obs_shapes = self._calc_obs_shapes()
        state_shape = self._calc_state_shape()
        observation_space_list = [Box(low=0., high=2., shape=team_obs_shapes[j], dtype=np.float32) for j in range(len(team_sizes)) for i in range(team_sizes[j])]

        self.state_space = Box(low=0., high=2., shape=state_shape, dtype=np.float32)
        reward_low, reward_high = reward_range

        if extra_features:
            for space in observation_space_list:
                idx = space.shape[2] - 3 if minimap_mode else space.shape[2] - 1
                space.low[:, :, idx] = reward_low
                space.high[:, :, idx] = reward_high
            idx_state = self.state_space.shape[2] - 3 if minimap_mode else self.state_space.shape[2] - 1
            self.state_space.low[:, :, idx_state] = reward_low
            self.state_space.high[:, :, idx_state] = reward_high

        self.action_spaces = {agent: space for agent, space in zip(self.agents, action_spaces_list)}
        self.observation_spaces = {agent: space for agent, space in zip(self.agents, observation_space_list)}

        self._zero_obs = {agent: np.zeros_like(space.low) for agent, space in self.observation_spaces.items()}
        self.base_state = np.zeros(self.state_space.shape)
        walls = self.env._get_walls_info()
        wall_x, wall_y = zip(*walls)
        self.base_state[wall_x, wall_y, 0] = 1
        self._renderer = None
        self.frames = 0

    def seed(self, seed=None):
        if seed is None:
            seed = seeding.create_seed(seed, max_bytes=4)
        self.env.set_seed(seed)

    def _calc_obs_shapes(self):
        view_spaces = [self.env.get_view_space(handle) for handle in self.handles]
        feature_spaces = [self.env.get_feature_space(handle) for handle in self.handles]
        assert all(len(tup) == 3 for tup in view_spaces)
        assert all(len(tup) == 1 for tup in feature_spaces)
        feat_size = [[fs[0]] for fs in feature_spaces]
        for feature_space in feat_size:
            if not self.extra_features:
                feature_space[0] = 2 if self.minimap_mode else 0
        obs_spaces = [(view_space[:2] + (view_space[2] + feature_space[0],)) for view_space, feature_space in zip(view_spaces, feat_size)]
        return obs_spaces

    def _calc_state_shape(self):
        feature_spaces = [self.env.get_feature_space(handle) for handle in self._all_handles]

        # map channel and agent pair channel. Remove global agent position when minimap mode and extra features
        state_depth = (max(feature_spaces)[0] - 2) * self.extra_features + 1 + len(self._all_handles) * 2

        return (self.map_size, self.map_size, state_depth)

    def render(self, mode="human"):
        if self._renderer is None:
            self._renderer = Renderer(self.env, self.map_size, mode)
        assert mode == self._renderer.mode, "mode must be consistent across render calls"
        return self._renderer.render(mode)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def reset(self):
        self.agents = self.possible_agents[:]
        self.env.reset()
        self.frames = 0
        self.all_dones = {agent: False for agent in self.possible_agents}
        self.generate_map()
        return self._observe_all()

    def _observe_all(self):
        observes = [None] * self.max_num_agents
        for handle in self.handles:
            ids = self.env.get_agent_id(handle)
            view, features = self.env.get_observation(handle)

            if self.minimap_mode and not self.extra_features:
                features = features[:, -2:]
            if self.minimap_mode or self.extra_features:
                feat_reshape = np.expand_dims(np.expand_dims(features, 1), 1)
                feat_img = np.tile(feat_reshape, (1, view.shape[1], view.shape[2], 1))
                fin_obs = np.concatenate([view, feat_img], axis=-1)
            else:
                fin_obs = np.copy(view)
            for id, obs in zip(ids, fin_obs):
                observes[id] = obs

        ret_agents = set(self.agents)
        return {agent: obs if obs is not None else self._zero_obs[agent] for agent, obs in zip(self.possible_agents, observes) if agent in ret_agents}

    def _all_rewards(self):
        rewards = np.zeros(self.max_num_agents)
        for handle in self.handles:
            ids = self.env.get_agent_id(handle)
            rewards[ids] = self.env.get_reward(handle)
        ret_agents = set(self.agents)
        return {agent: float(rew) for agent, rew in zip(self.possible_agents, rewards) if agent in ret_agents}

    def _all_dones(self, step_done=False):
        dones = np.ones(self.max_num_agents, dtype=np.bool)
        if not step_done:
            for handle in self.handles:
                ids = self.env.get_agent_id(handle)
                dones[ids] = ~self.env.get_alive(handle)
        ret_agents = set(self.agents)
        return {agent: bool(done) for agent, done in zip(self.possible_agents, dones) if agent in ret_agents}

    def state(self):
        '''
        Returns an observation of the global environment
        '''
        state = np.copy(self.base_state)

        for handle in self._all_handles:
            view, features = self.env.get_observation(handle)

            pos = self.env.get_pos(handle)
            pos_x, pos_y = zip(*pos)
            state[pos_x, pos_y, 1 + handle.value * 2] = 1
            state[pos_x, pos_y, 2 + handle.value * 2] = view[:, view.shape[1] // 2, view.shape[2] // 2, 2]

            if self.extra_features:
                add_zeros = np.zeros((features.shape[0], state.shape[2] - (1 + len(self.team_sizes) * 2 + features.shape[1])))

                rewards = features[:, -1]
                actions = features[:, :-1]
                actions = np.concatenate((actions, add_zeros), axis=1)
                rewards = rewards.reshape(len(rewards), 1)
                state_features = np.hstack((actions, rewards))

                state[pos_x, pos_y, 1 + len(self.team_sizes) * 2:] = state_features
        return state

    def step(self, all_actions):
        """
        :param all_actions:  this should be a dict with self.possible_agents as the key and int action value
        :return:
        """
        action_list = [0] * self.max_num_agents
        self.agents = [agent for agent in self.agents if not self.all_dones[agent]]
        self.env.clear_dead()
        for i, agent in enumerate(self.possible_agents):
            if agent in all_actions:
                action_list[i] = all_actions[agent]
        all_actions = np.asarray(action_list, dtype=np.int32)
        start_point = 0
        for i in range(len(self.handles)):
            size = self.team_sizes[i]
            self.env.set_action(self.handles[i], all_actions[start_point:(start_point + size)])
            start_point += size

        self.frames += 1
        done = self.env.step() or self.frames >= self.max_cycles

        all_infos = {agent: {} for agent in self.agents}
        all_dones = self._all_dones(done)
        all_rewards = self._all_rewards()
        all_observes = self._observe_all()
        self.all_dones = all_dones
        return all_observes, all_rewards, all_dones, all_infos



from pettingzoo.utils.conversions import from_parallel_wrapper
from gym.utils import EzPickle


default_map_size = 45
max_cycles_default = 1000
KILL_REWARD = 5
minimap_mode_default = False
default_reward_args = dict(step_reward=-0.005, dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2)


def parallel_env(map_size=default_map_size, max_cycles=max_cycles_default, minimap_mode=minimap_mode_default, extra_features=False, **reward_args):
    env_reward_args = dict(**default_reward_args)
    env_reward_args.update(reward_args)
    return _parallel_env(map_size, minimap_mode, env_reward_args, max_cycles, extra_features)


def raw_env(map_size=default_map_size, max_cycles=max_cycles_default, minimap_mode=minimap_mode_default, extra_features=False, **reward_args):
    return from_parallel_wrapper(parallel_env(map_size, max_cycles, minimap_mode, extra_features, **reward_args))


#env = make_env(raw_env)


def get_config(map_size, minimap_mode, step_reward, dead_penalty, attack_penalty, attack_opponent_reward):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": minimap_mode})
    cfg.set({"embedding_size": 10})

    options = {
        'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
        'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(1.5),
        'damage': 2, 'kill_reward': KILL_REWARD, 'step_recover': 0.1,
        'step_reward': step_reward, 'dead_penalty': dead_penalty, 'attack_penalty': attack_penalty
    }
    small = cfg.register_agent_type(
        "small",
        options
    )

    g0 = cfg.add_group(small)
    g1 = cfg.add_group(small)

    a = gw.AgentSymbol(g0, index='any')
    b = gw.AgentSymbol(g1, index='any')

    # reward shaping to encourage attack
    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=attack_opponent_reward)
    cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=b, value=attack_opponent_reward)

    return cfg


class _parallel_env(magent_parallel_env, EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array'], 'name': "battle_v3"}

    def __init__(self, map_size, minimap_mode, reward_args, max_cycles, extra_features):
        EzPickle.__init__(self, map_size, minimap_mode, reward_args, max_cycles, extra_features)
        assert map_size >= 12, "size of map must be at least 12"
        env = magent.GridWorld(get_config(map_size, minimap_mode, **reward_args), map_size=map_size)
        self.leftID = 0
        self.rightID = 1
        reward_vals = np.array([KILL_REWARD] + list(reward_args.values()))
        reward_range = [np.minimum(reward_vals, 0).sum(), np.maximum(reward_vals, 0).sum()]
        names = ["red", "blue"]
        super().__init__(env, env.get_handles(), names, map_size, max_cycles, reward_range, minimap_mode, extra_features)

    def generate_map(self):
        env, map_size, handles = self.env, self.map_size, self.handles
        """ generate a map, which consists of two squares of agents"""
        width = height = map_size
        init_num = map_size * map_size * 0.04
        gap = 3

        #self.leftID, self.rightID = self.rightID, self.leftID           #switching initial position

        # left
        n = init_num
        side = int(math.sqrt(n)) * 2
        pos = []
        for x in range(width // 2 - gap - side, width // 2 - gap - side + side, 2):
            for y in range((height - side) // 2, (height - side) // 2 + side, 2):
                if 0 < x < width - 1 and 0 < y < height - 1:
                    pos.append([x, y, 0])
        team1_size = len(pos)
        env.add_agents(handles[self.leftID], method="custom", pos=pos)

        # right
        n = init_num
        side = int(math.sqrt(n)) * 2
        pos = []
        for x in range(width // 2 + gap, width // 2 + gap + side, 2):
            for y in range((height - side) // 2, (height - side) // 2 + side, 2):
                if 0 < x < width - 1 and 0 < y < height - 1:
                    pos.append([x, y, 0])

        pos = pos[:team1_size]
        env.add_agents(handles[self.rightID], method="custom", pos=pos)


def count_alive(env, n_red, n_blue):
    red_handle, blue_handle = env.env.get_handles()
    red_alive, blue_alive = 0, 0

    a = env.env.get_alive(red_handle)
    red_alive = np.count_nonzero(a + np.zeros(len(a)))

    b = env.env.get_alive(blue_handle)
    blue_alive = np.count_nonzero(b + np.zeros(len(b)))

    return n_red - red_alive, n_blue - blue_alive

def save_models(save_path, model_list, model_name, updates, save_optim=True):
    save_dict = {}
    for i in range(len(model_name)):
        save_dict[model_name[i]] = model_list[model_name[i]].critic.state_dict()
        if save_optim:
            save_dict[model_name[i]+'_optim'] = model_list[model_name[i]].critic_optim.state_dict()

    torch.save(save_dict, save_path + '/battle_models{}.pth'.format(updates))
    print('saved')

def load_models(load_path, model_list, model_name, updates, load_optim):
    load_dict = torch.load(load_path + '/battle_models{}.pth'.format(updates))
    for i in range(len(model_name)):
        model_list[model_name[i]].critic.load_state_dict(load_dict[model_name[i]])
        if load_optim:
            model_list[model_name[i]].critic_optim.load_state_dict(load_dict[model_name[i]+'_optim'])

    print('Loaded')
    return model_list


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

#########################################  Models  ##################################################


def weights_init_kaiming(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def weights_init_xavier(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class CNN_encoder(nn.Module):
    """
    view size : [batch, view_space]

    """

    def __init__(self):
        super(CNN_encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            Flatten()
        )

    def forward(self, view_state):
        """
        [batch, 10,10,5]
        """
        state = view_state.permute(0, 3, 1, 2)

        return self.net(state)  # [batch, 128]


class Critic(nn.Module):
    def __init__(self, output_size, input_size=128, hidden_size=128, if_feature=False, feature_size=None):
        super().__init__()

        self.conv = CNN_encoder()

        self.if_feature = if_feature

        self.input_size = input_size
        self.output_size = output_size

        if if_feature:
            self.view_linear = nn.Linear(input_size, hidden_size)
            self.feature_linear = nn.Linear(feature_size, feature_size)
            self.concat_linear = nn.Sequential(
                nn.Linear(hidden_size + feature_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        else:
            self.view_linear = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )

        self.apply(weights_init_kaiming)

    def forward(self, view_state, feature_state=None):
        """
        view_state: [batch, 10,10,5]
        feature_state: [batch, 34] or None
        """
        if self.if_feature:
            view_encoded_state = self.view_linear(self.conv(view_state))
            view_encoded_state = F.relu(view_encoded_state)

            feature_encoded_state = F.relu(self.feature_linear(feature_state))
            concate = torch.cat([view_encoded_state, feature_encoded_state], dim=-1)  # []batch, hidden+feature
            return self.concat_linear(concate)  # [batch, output_size]

        else:
            conv_encoded = self.conv(view_state)  # [batch, 128]
            # print('conv encoded shape', conv_encoded.shape)
            return self.view_linear(conv_encoded)


class DQN_IL(object):
    def __init__(self, obs_dim, action_space, epsilon=1, max_episode=500):
        """

        """

        self.lr = 0.0003
        self.gamma = 0.95
        # self.single_handle = single_handle
        self.target_replace_iter = 1

        self.obs_dim = obs_dim
        self.action_space = action_space

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.critic = Critic(output_size=self.action_space.n).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.critic_target = Critic(output_size=self.action_space.n).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.eps = epsilon
        self.eps_end = 0.05
        self.eps_delay = 1 / (max_episode * 100)  # after 10 round of training

    def choose_action(self, state):
        """
        state: [13,13,5]
        """
        self.eps = max(self.eps_end, self.eps - self.eps_delay)
        if random.random() < self.eps:

            action = np.array([self.action_space.sample()])
        else:

            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            critic = self.critic(state)
            action = critic.cpu().detach().max(1)[1].numpy()

        return action

    def learn(self, memory, batch_size, updates):

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)  # [batch, view_space]
        action_batch = torch.FloatTensor(action_batch).to(self.device)  # [batch, 1]
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(-1).to(self.device)  # [batch, 1]
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)  # [batch, view_space]
        mask_batch = torch.FloatTensor(mask_batch).unsqueeze(-1).to(self.device)  # [batch, 1]

        batch_current_q = self.critic(state_batch).gather(1, action_batch.long())  # [batch, 1]
        batch_next_q = self.critic_target(next_state_batch).detach()  # [batch, action_dim]
        batch_next_max_q = batch_next_q.max(1)[0].unsqueeze(-1)  # [batch, 1]
        batch_target = reward_batch + self.gamma * mask_batch * batch_next_max_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(batch_current_q, batch_target)

        self.critic_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optim.step()

        if updates % self.target_replace_iter == 0:
            self.critic_target.load_state_dict(self.critic.state_dict())

        return loss.item()




parser = argparse.ArgumentParser(description='PyTorch Independent-Q on battle games Args')
parser.add_argument('--map_size', type = int, default = 15)
parser.add_argument('--max_cycles', type = int, default = 500)
parser.add_argument('--env_seed', type = int,  default = 1234)
parser.add_argument('--render', type = bool, default = True)
parser.add_argument('--single_handle', type = bool, default = True)
parser.add_argument('--print_interval', type = int, default = 500)
parser.add_argument('--num_steps', type = int, default = 1000000)

parser.add_argument('--memory_size', type = int, default = 500000)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--train_interval', type = int, default = 1)
parser.add_argument('--train_per_step', type = int, default = 1)
parser.add_argument('--saving', type = bool, default = True)

parser.add_argument('--eval_interval', type = int, default = 5),
parser.add_argument('--eval_episode', type = int, default = 20)
parser.add_argument('--tensorboard', type = bool, default = True)
args = parser.parse_args('')


map_size = args.map_size
seed = args.env_seed

env = parallel_env(map_size = map_size, max_cycles = args.max_cycles)       #battle_v3.parallel_env(map_size=map_size, max_cycles=500)
env.seed(seed)
num_red = 3
num_blue = 3
save_model_path = os.getcwd() + '/battle_IL'
key_list = env.agents

if_render = args.render
if_single_handle = args.single_handle
print_interval = args.print_interval

# battle_model = DQN(env, if_single_handle)
models = {}
replay_memories = {}
for i in env.agents:
    if 'red' in i:
        obs_dim = env.observation_spaces[i]
        action_space = env.action_spaces[i]
        models[i] = DQN_IL(obs_dim, action_space)

        replay_memories[i] = ReplayMemory(capacity=args.memory_size, seed=args.env_seed)
    # models.append(DQN_IL(obs_dim, action_space))

#save_models(save_model_path, models, key_list[:3], 123, save_optim=True)
#models = load_models(save_model_path, models, key_list[:3], 123, load_optim = True)
if args.tensorboard:
    writer = SummaryWriter('runs/{}_IL_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'Battle',
                                                             args.map_size, if_single_handle))


batch_size = args.batch_size
updates = 0
train_interval = args.train_interval
train_per_step = args.train_per_step
eval_interval = args.eval_interval
eval_episode = args.eval_episode
explore_eps = 0.2

# red_buffer = ReplayMemory(capacity = 500000, seed = seed)
# blue_buffer = ReplayMemory(capacity = 500000, seed = seed)
train_loss_list = []
train_red_reward = []
train_blue_reward = []

test_red_reward = []
test_blue_reward = []
# red_idx_array = env.env.get_agent_id(env.handles[0])
# blue_idx_array = env.env.get_agent_id(env.handles[1])

total_numsteps = 0
num_steps = 100000

max_killed = 0

# action_space = env.action_spaces

for i_episode in itertools.count(1):

    done = False
    episode_red_reward = 0
    episode_blue_reward = 0

    episode_steps = 0
    state = env.reset()

    while not done:

        key_list = env.agents  # alive agent list
        action_dict = {}
        for agent in key_list:
            if 'red' in agent:
                action_dict[agent] = models[agent].choose_action(state[agent])
            elif 'blue' in agent:
                action_dict[agent] = np.array([env.action_spaces[agent].sample()])  # single handle
            else:
                raise NotImplementedError

        # action_dict = {key_list[i]: np.array([action_space[key_list[i]].sample()]) for i in range(len(key_list))}
        # action_dict = battle_model.choose_action(state, eps = explore_eps)

        next_state, reward_dict, done_dict, _ = env.step(action_dict)

        done = all(done_dict.values())  # normally the done_dict tells whether the agents are ~alive, or all are True
        # if maximum episode is reach or either team is eliminated (told by env.env)

        killed_red, killed_blue = count_alive(env, num_red, num_blue)

        temp_red_reward, temp_blue_reward = 0, 0
        # num_red_reward, num_blue_reward = 0, 0
        for key, value in reward_dict.items():
            if 'red' in key:
                # red_buffer.push(state[key], action_dict[key], reward_dict[key], next_state[key],
                #                float(not done_dict[key]))
                replay_memories[key].push(state[key], action_dict[key], reward_dict[key], next_state[key],
                                          float(not done_dict[key]))

                temp_red_reward += value
                # num_red_reward += 1
            elif 'blue' in key:
                # if not if_single_handle:
                #    blue_buffer.push(state[key], action_dict[key], reward_dict[key], next_state[key],
                #                float(not done_dict[key]))

                temp_blue_reward += value
                # num_blue_reward += 1
            else:
                raise NotImplementedError

        #average_red_reward = temp_red_reward  # /num_red_reward    #averaged over agents
        #average_blue_reward = temp_blue_reward  # /num_blue_reward

        episode_red_reward += temp_red_reward   #average_red_reward
        episode_blue_reward +=  temp_blue_reward  #average_blue_reward

        train_red_reward.append(episode_red_reward)
        train_blue_reward.append(episode_blue_reward)

        if len(replay_memories['red_0']) > batch_size and total_numsteps % train_interval == 0:
            temp_loss = 0
            for _ in range(train_per_step):

                for agent in env.possible_agents:
                    if 'red' in agent:
                        agent_loss = models[agent].learn(replay_memories[agent], batch_size, updates)
                        temp_loss += agent_loss

                if args.tensorboard:
                    writer.add_scalar('loss/train (red)', temp_loss, updates)
                #        temp_loss = battle_model.learn(red_buffer, batch_size, updates)

                # print('Training loss = ', temp_loss)
                updates += 1
            train_loss_list.append(temp_loss / train_per_step)
            # print('trained')

        if if_render:
            env.render()

        state = next_state
        episode_steps += 1
        total_numsteps += 1
        # print('episode_steps', episode_steps)

        if episode_steps % print_interval == 0 or done:
            print('Episode {}; {} red killed, reward = {:.3f}; {} blue killed, reward = {:.2f}'.format(episode_steps,
                                                                                                       killed_red,
                                                                                                       episode_red_reward,
                                                                                                       killed_blue,
                                                                                                       episode_blue_reward))
    # print('Episode red reward = {:.5f}, Episode blue reward = {:.5f}'.format(episode_red_reward/episode_steps,
    #                                                                episode_blue_reward/episode_steps))

    if total_numsteps > num_steps:
        break
    if args.tensorboard:
        writer.add_scalar('reward/train (red)', episode_red_reward, i_episode)
        writer.add_scalar('reward/train (blue)', episode_blue_reward, i_episode)
        writer.add_scalar('killed/train (blue)', killed_blue, i_episode)


    if i_episode % eval_interval == 0:
        average_killed_red = 0
        average_killed_blue = 0
        average_reward_red = 0
        average_reward_blue = 0

        for i in range(eval_episode):
            done = False
            episode_red_reward = 0
            episode_blue_reward = 0

            episode_red_killed = 0
            episode_blue_killed = 0

            episode_steps = 0
            state = env.reset()

            while not done:
                # action_dict = {key_list[i]: np.array([action_space[key_list[i]].sample()]) for i in range(len(key_list))}
                # action_dict = battle_model.choose_action(state, eps = 0)
                key_list = env.agents  # alive agent list
                action_dict = {}
                for agent in key_list:
                    if 'red' in agent:
                        action_dict[agent] = models[agent].choose_action(state[agent])
                    elif 'blue' in agent:
                        action_dict[agent] = np.array([env.action_spaces[agent].sample()])  # single handle
                    else:
                        raise NotImplementedError

                next_state, reward_dict, done_dict, _ = env.step(action_dict)

                done = all(done_dict.values())

                temp_red_reward, temp_blue_reward = 0, 0
                for key, value in reward_dict.items():
                    if 'red' in key:
                        temp_red_reward += value
                    elif 'blue' in key:
                        temp_blue_reward += value
                    else:
                        raise NotImplementedError
                average_reward_red += temp_red_reward
                average_reward_blue += temp_blue_reward

                state = next_state

            killed_red, killed_blue = count_alive(env, num_red, num_blue)

            average_killed_red += killed_red
            average_killed_blue += killed_blue
        average_killed_red /= eval_episode
        average_killed_blue /= eval_episode

        average_reward_red /= eval_episode
        average_reward_blue /= eval_episode
        test_red_reward.append(average_reward_red)
        test_blue_reward.append(average_reward_blue)

        print('--------------------------------------------------------------------------------------------')
        print('Testing: {} red killed, reward = {} ; {} blue killed, reward = {}'.format(average_killed_red,
                                                                                         average_reward_red,
                                                                                         average_killed_blue,
                                                                                         average_reward_blue))
        if args.tensorboard:
            writer.add_scalar('reward/test (red)', average_reward_red, i_episode)
            writer.add_scalar('reward/test (blue)', average_reward_blue, i_episode)
            writer.add_scalar('killed/test (blue)', average_killed_blue, i_episode)



        if average_killed_blue > max_killed and args.saving:
            save_models(save_model_path, models, key_list[:3], updates)
            max_killed = average_killed_blue

        print('--------------------------------------------------------------------------------------------')








