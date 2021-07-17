

import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
import itertools

from algo.independent_Q import DQN_IL
from env.battle_env import parallel_env
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Independent-Q on battle games Args')
parser.add_argument('--map_size', type = int, default = 20)
parser.add_argument('--max_cycles', type = int, default = 500)
parser.add_argument('--env_seed', type = int,  default = 1234)
parser.add_argument('--render', type = bool, default = True)
parser.add_argument('--single_handle', type = bool, default = False)
parser.add_argument('--print_interval', type = int, default = 500)
parser.add_argument('--num_steps', type = int, default = 100000)

parser.add_argument('--memory_size', type = int, default = 500000)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--train_interval', type = int, default = 1)
parser.add_argument('--train_per_step', type = int, default = 1)
parser.add_argument('--saving', type = bool, default = False)

parser.add_argument('--eval_interval', type = int, default = 5),
parser.add_argument('--eval_episode', type = int, default = 20)
parser.add_argument('--tensorboard', type = bool, default = True)
args = parser.parse_args('')

##########################################  env setup  ##########################################
env = parallel_env(map_size = args.map_size, max_cycles = args.max_cycles)       #battle_v3.parallel_env(map_size=map_size, max_cycles=500)
env.seed(args.env_seed)
num_red = len(env.agents)/2
num_blue = num_red
save_model_path = os.getcwd() + '/battle_IL'
key_list = env.agents

##########################################  model setup  ##########################################
models = {}
replay_memories = {}
for i in env.agents:
    obs_dim = env.observation_spaces[i]
    action_space = env.action_spaces[i]

    if 'red' in i:
        models[i] = DQN_IL(obs_dim, action_space)
        replay_memories[i] = ReplayMemory(capacity=args.memory_size, seed=args.env_seed)
    elif 'blue' in i:
        if not args.single_handle:
            models[i] = DQN_IL(obs_dim, action_space)
            replay_memories[i] = ReplayMemory(capacity=args.memory_size, seed=args.env_seed+1)
    else:
        raise NotImplementedError


##########################################  train-test iteration  ##########################################
if args.tensorboard:
    writer = SummaryWriter('runs/{}_IL_{}_{}_single-agent: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'Battle',args.map_size, args.single_handle))

updates = 0

if args.single_handle:
    red_train_loss_list,train_red_reward,train_blue_reward = [], [], []
else:
    red_train_loss_list, blue_train_loss_list, train_red_reward,train_blue_reward = [], [], [], []

test_red_reward,test_blue_reward = [], []

total_numsteps = 0
max_killed = 0

# action_space = env.action_spaces


for i_episode in itertools.count(1):

    done = False
    episode_red_reward = 0
    episode_blue_reward = 0

    episode_steps = 0
    state = env.reset()

    while not done:

        ##############################################  Taking actions    ##############################################
        key_list = env.agents  # alive agent list
        action_dict = {}
        for agent in key_list:
            if not args.single_handle:
                action_dict[agent] = models[agent].choose_action(state[agent])
            else:

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

        ##############################################  Recording data   ##############################################
        temp_red_reward, temp_blue_reward = 0, 0
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
                if not args.single_handle:
                    replay_memories[key].push(state[key], action_dict[key], reward_dict[key], next_state[key],
                                              float(not done_dict[key]))

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


############################################## Model training   ##############################################
        if len(replay_memories['red_0']) > args.batch_size and total_numsteps % args.train_interval == 0:
            if args.single_handle:
                red_temp_loss = 0
            else:
                red_temp_loss, blue_temp_loss = 0, 0

            for _ in range(args.train_per_step):

                for agent in env.possible_agents:
                    if 'red' in agent:
                        agent_loss = models[agent].learn(replay_memories[agent], args.batch_size, updates)
                        red_temp_loss += agent_loss
                    elif ('blue' in agent) and (not args.single_handle):
                        agent_loss = models[agent].learn(replay_memories[agent], args.batch_size, updates)
                        blue_temp_loss += agent_loss
                    else:
                        pass

                if args.tensorboard:
                    writer.add_scalar('loss/train (red)', red_temp_loss, updates)
                    if not args.single_handle:
                        writer.add_scalar('loss/train (blue)', blue_temp_loss, updates)
                #        temp_loss = battle_model.learn(red_buffer, batch_size, updates)

                # print('Training loss = ', temp_loss)
                updates += 1
            red_train_loss_list.append(red_temp_loss / args.train_per_step)
            if not args.single_handle:
                blue_train_loss_list.append(blue_temp_loss/args.train_per_step)
            # print('trained')

        if args.render:
            env.render()

        state = next_state
        episode_steps += 1
        total_numsteps += 1
        # print('episode_steps', episode_steps)

        if episode_steps % args.print_interval == 0 or done:
            if done and episode_steps < args.max_cycles:
                if killed_blue == num_blue:
                    print_line = 'red win'
                elif killed_red == num_red:
                    print_line = 'blue win'
                else:
                    raise ValueError
            else:
                print_line = 'fail'

            print('Iteration {}, episode length {}; {} red killed, reward = {:.3f}; {} blue killed, reward = {:.2f}. {}.'.format(i_episode,
                                                                                                                            episode_steps,
                                                                                                                            killed_red,
                                                                                                                            episode_red_reward,
                                                                                                                            killed_blue,
                                                                                                                            episode_blue_reward,
                                                                                                                            print_line))
    # print('Episode red reward = {:.5f}, Episode blue reward = {:.5f}'.format(episode_red_reward/episode_steps,
    #                                                                episode_blue_reward/episode_steps))

    if total_numsteps > args.num_steps:
        break
    if args.tensorboard:
        writer.add_scalar('train reward/red', episode_red_reward, i_episode)
        writer.add_scalar('train killed/red', killed_red, i_episode)
        writer.add_scalar('train reward/blue', episode_blue_reward, i_episode)
        writer.add_scalar('train killed/blue', killed_blue, i_episode)


    #################################  Model testing   ##############################################

    if i_episode % args.eval_interval == 0:
        average_killed_red = 0
        average_killed_blue = 0
        average_reward_red = 0
        average_reward_blue = 0
        num_success = 0

        for i in range(args.eval_episode):
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
                    if not args.single_handle:
                        action_dict[agent] = models[agent].choose_action(state[agent])
                    else:

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
                episode_steps += 1

            killed_red, killed_blue = count_alive(env, num_red, num_blue)

            average_killed_red += killed_red
            average_killed_blue += killed_blue
            if episode_steps < args.max_cycles:
                num_success += 1

        average_killed_red /= args.eval_episode
        average_killed_blue /= args.eval_episode

        average_reward_red /= args.eval_episode
        average_reward_blue /= args.eval_episode
        test_red_reward.append(average_reward_red)
        test_blue_reward.append(average_reward_blue)

        success_rate = num_success / args.eval_episode

        print('--------------------------------------------------------------------------------------------')
        print('Testing: {} red killed, reward = {} ; {} blue killed, reward = {}; Success rate = {}'.format(average_killed_red,
                                                                                         average_reward_red,
                                                                                         average_killed_blue,
                                                                                         average_reward_blue,
                                                                                         success_rate))
        if args.tensorboard:
            writer.add_scalar('test reward/red', average_reward_red, i_episode)
            writer.add_scalar('test killed/red', average_killed_red, i_episode)
            writer.add_scalar('test reward/blue', average_reward_blue, i_episode)
            writer.add_scalar('test killed/blue', average_killed_blue, i_episode)
            writer.add_scalar('test success rate', success_rate, i_episode)



        if average_killed_blue > max_killed and args.saving:

            save_key_list = key_list[:3] if args.single_handle else key_list

            save_models(save_model_path, models, key_list[:3], updates)
            max_killed = average_killed_blue

        print('--------------------------------------------------------------------------------------------')






