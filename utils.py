import numpy as np
import random
import torch

def count_alive(env, n_red, n_blue):
    red_handle, blue_handle = env.env.get_handles()
    red_alive, blue_alive = 0, 0

    a = env.env.get_alive(red_handle)
    red_alive = np.count_nonzero(a + np.zeros(len(a)))

    b = env.env.get_alive(blue_handle)
    blue_alive = np.count_nonzero(b + np.zeros(len(b)))

    return n_red - red_alive, n_blue - blue_alive


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


def save_models(save_path, model_list, model_name, updates, save_optim):
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


attack_dict = {(5, 5): 13, (6, 5): 14, (7, 5): 15, (5, 6): 16, (7, 6): 17, (5, 7): 18, (6, 7): 19, (7, 7): 20}  # coord
attack_coord = list(attack_dict.keys())

def blue_policy(state, action_space):
    """
    state: [13,13,5]
    """
    opponent_state = state[:, :, 3]
    action_list = []
    for i in attack_coord:
        coord1, coord2 = i
        if opponent_state[coord2, coord1] > 0:
            action_list.append(attack_dict[i])
    #print(action_list)
    if len(action_list) == 0:
        return np.array(action_space.sample())
    else:
        return np.random.choice(action_list, 1)






