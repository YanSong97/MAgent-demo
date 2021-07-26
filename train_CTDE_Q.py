from algo.CTDE_Q import CTDE_Q
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
import itertools
from env.battle_env import parallel_env
from utils import *



parser = argparse.ArgumentParser(description='PyTorch CTDE-Q on battle games Args')
parser.add_argument('--map_size', type = int, default = 15)
parser.add_argument('--max_cycles', type = int, default = 500, help = 'maximun length of each episode')
parser.add_argument('--seed_maxsize', type = int,  default = 30, help = 'randly set the seed at each episode')
parser.add_argument('--shuffle_init', type = bool, default = False, help='whether to randomly shuffle the postion of agents in the same team')
parser.add_argument('--render', type = bool, default = True)
parser.add_argument('--single_handle', type = bool, default = True, help = 'if True, only model one team')
parser.add_argument('--advanced_policy', type = bool, default = False, help ='if true, the blue team has a fixed policy, attacking opponent within randge or act randomlt')
parser.add_argument('--print_interval', type = int, default = 500)
parser.add_argument('--num_steps', type = int, default = 1000000)

parser.add_argument('--memory_size', type = int, default = 500000)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--train_interval', type = int, default = 1)
parser.add_argument('--train_per_step', type = int, default = 1)
parser.add_argument('--saving', type = bool, default = False)

parser.add_argument('--eval_interval', type = int, default = 5),
parser.add_argument('--eval_episode', type = int, default = 20)
parser.add_argument('--tensorboard', type = bool, default = True)
args = parser.parse_args('')

env = parallel_env(map_size = args.map_size, max_cycles = args.max_cycles,shuffle_init=args.shuffle_init)       #battle_v3.parallel_env(map_size=map_size, max_cycles=500)
env.seed(random.randint(0, args.seed_maxsize))
num_red, num_blue = len(env.agents)/2,len(env.agents)/2
save_model_path = os.getcwd() + '/save_fixed'
key_list = env.agents
print(len(key_list))

models = {}
replay_memories = {}
for i in env.agents:
    obs_dim = env.observation_spaces[i]
    action_space = env.action_spaces[i]
    if 'red' in i:
        models[i] = CTDE_Q(obs_dim, action_space, len(env.agents) / 2)      #single handle
        replay_memories[i] = ReplayMemory(capacity=args.memory_size, seed=random.randint(0, 30))

updates = 0
total_numsteps = 0

if args.tensorboard:
    writer = SummaryWriter('runs/{}_CTDE-Q_{}_{}_single-agent: {}, agent_num {}, random init {}.'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                                    'Battle',args.map_size, args.single_handle,
                                                                                    len(key_list),
                                                                                    args.shuffle_init))


for i_episode in itertools.count(1):

    done = False
    episode_red_reward = 0
    episode_blue_reward = 0

    episode_steps = 1
    env.seed(random.randint(0, args.seed_maxsize))
    state = env.reset()

    while not done:

        ##############################################  Taking actions    ##############################################
        current_key_list = env.agents  # alive agent list
        action_dict = {}
        for key in key_list:
            if key not in current_key_list:   #agent has died
                action_dict[key] = np.array([6])
            else:
                if 'red' in key:          #
                    action_dict[key] = models[key].choose_action(state[key], evaluate=False).numpy()
                elif 'blue' in key:
                    action_dict[key] = np.array([env.action_spaces[key].sample()])
                else:
                    raise NotImplementedError


        next_state, reward_dict, done_dict, _ = env.step(action_dict)
        done = all(done_dict.values())
        killed_red, killed_blue = count_alive(env, num_red, num_blue)

        all_red_action = np.concatenate(list(action_dict.values()))[:int(len(key_list)/2)]

        ##############################################  recording data   ##############################################
        temp_red_reward, temp_blue_reward = 0, 0
        for key, value in reward_dict.items():
            if 'red' in key:
                replay_memories[key].push(state[key], all_red_action, reward_dict[key], next_state[key],
                                          float(not done_dict[key]))

                temp_red_reward += value
                # num_red_reward += 1
            elif 'blue' in key:

                temp_blue_reward += value
                # num_blue_reward += 1
            else:
                raise NotImplementedError

        episode_red_reward += temp_red_reward   #average_red_reward
        episode_blue_reward +=  temp_blue_reward  #average_blue_reward

        if len(replay_memories['red_0']) > args.batch_size and total_numsteps % args.train_interval == 0:
            red_critic_loss, red_actor_loss = 0, 0
            for _ in range(args.train_per_step):
                agent_idx = 0
                for agent in env.possible_agents:
                    if 'red' in agent:
                        temp_critic_loss, temp_actor_loss = models[agent].learn(replay_memories[agent],
                                                                                args.batch_size, updates,
                                                                                actor_container=models,
                                                                                current_actor=agent,
                                                                                actor_idx=agent_idx)
                        if args.tensorboard:
                            writer.add_scalar('loss/train-critic', temp_critic_loss, updates)
                            writer.add_scalar('loss/train-actor', temp_actor_loss, updates)
                    else:
                        pass
                    agent_idx += 1

                    updates += 1

        if episode_steps % args.print_interval == 0 or done:
            print('iteration {}; episode length {}; red killed={}, reward = {:.3f}; '.format(i_episode,
                                                                                            episode_steps,
                                                                                            killed_red,
                                                                                            episode_red_reward) +
                  'blue_killed = {}, reward = {:.3f}'.format(killed_blue, episode_blue_reward))

        if args.render:
            env.render()


        state = next_state
        episode_steps += 1
        total_numsteps += 1


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
            env.seed(random.randint(0, args.seed_maxsize))
            state = env.reset()


            while not done:

                current_key_list = env.agents  # alive agent list
                action_dict = {}
                for key in key_list:
                    if key not in current_key_list:  # agent has died
                        action_dict[key] = np.array([6])
                    else:
                        if 'red' in key:  #
                            action_dict[key] = models[key].choose_action(state[key], evaluate=True,test=True).numpy()
                        elif 'blue' in key:
                            action_dict[key] = np.array([env.action_spaces[key].sample()])
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
        #test_red_reward.append(average_reward_red)
        #test_blue_reward.append(average_reward_blue)

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

        print('--------------------------------------------------------------------------------------------')




