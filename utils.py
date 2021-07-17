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

