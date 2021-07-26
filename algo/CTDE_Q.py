########  Centralised training decentralised execution DQN  ########################


from torch.optim import Adam

from .utils import *
from .Networks import *

attack_dict = {(5, 5): 13, (6, 5): 14, (7, 5): 15, (5, 6): 16, (7, 6): 17, (5, 7): 18, (6, 7): 19, (7, 7): 20}  # coord
attack_coord = list(attack_dict.keys())
def blue_policy(state, action_space):
    """
    state: [13,13,5]
    """
    opponent_state = state[:, :, 3]
    #print('opponent state shape', opponent_state.shape )
    action_list = []
    for i in attack_coord:
        coord1, coord2 = i
        if opponent_state[coord2, coord1] > 0:
            action_list.append(attack_dict[i])
    #print(action_list)
    if len(action_list) == 0:
        return np.array([action_space.sample()])
    else:
        return np.random.choice(action_list, 1)


class CTDE_Q(object):
    """
    each agent has its own policy and critic which takes others' actions

    """
    def __init__(self, obs_dim, action_space, num_agents):

        self.lr = 0.0003
        self.gamma = 0.95
        self.tau = 0.005
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.device = 'cpu'
        self.target_replace_iter = 1
        self.if_onehot = False

        self.critic = centralised_critic(num_agents = num_agents, action_size = self.action_space.n).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.lr)
        self.critic_target = centralised_critic(num_agents = num_agents, action_size = self.action_space.n).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.actor = Actor(action_dim=self.action_space.n).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr = self.lr)

        self.eps = 1.
        self.eps_end = 0.05
        self.eps_delay = 1 / (500 * 100)  # after 10 round of training

    def choose_action(self, state, evaluate, test=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if test:
            action = self.actor.sample(state_tensor, greedy=True)
            return action

        self.eps = max(self.eps_end, self.eps - self.eps_delay)
        if random.random() < self.eps:
            action = blue_policy(state, self.action_space)
            action = torch.tensor(action)
        else:
            if not evaluate:
                action = self.actor.sample(state_tensor,greedy = False)
            else:
                action = self.actor.sample(state_tensor, greedy = True)
        return action

        #state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        #self.eps = max(self.eps_end, self.eps - self.eps_delay)
        #if random.random() < self.eps:

            #action = torch.tensor([self.action_space.sample()])
        #    action = torch.tensor(np.random.choice(np.arange(13,21), 1))
        #else:

        #    if not evaluate:
        #        action = self.actor.sample(state,greedy = False)
        #    else:
        #        action = self.actor.sample(state, greedy = True)
        #action = blue_policy(state, self.action_space)

        #action = torch.tensor([action])
        #return torch.tensor(action)

    def learn(self, memory, batch_size, updates, actor_container, current_actor, actor_idx):
        """
        :param memory:  the stored action contains actions from other agents, may need to one-hot the actions
        :param actor_container: a dict of all the actor
        :param current_actor:  name of current actor
        :param actor_idx: the index of current actor
        :return:
        """
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)  # [batch, view_space]
        action_batch = torch.tensor(action_batch, dtype = int).to(self.device)  # [batch, num_actions]
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(-1).to(self.device)  # [batch, 1]
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)  # [batch, view_space]
        mask_batch = torch.FloatTensor(mask_batch).unsqueeze(-1).to(self.device)  # [batch, 1]

        #print(action_batch)

        one_hot_action = F.one_hot(action_batch, self.action_space.n).reshape(batch_size, -1)       #[batch_size, num_actions*action_dim]

        #critic loss
        #compute next action
        a_next = []
        with torch.no_grad():
            for agent in actor_container.keys():

                if agent == current_actor:
                    #print('next_state_batch shape', next_state_batch.shape)
                    #a_next.append(self.choose_action(next_state_batch, evaluate = True))         #[]
                    temp_action = self.actor.sample(next_state_batch, greedy = True)  #this can be replaced with a target actor
                    a_next.append(temp_action.unsqueeze(-1))
                else:
                    #a_next.append(actor_container[agent].choose_action(next_state_batch, evaluate=True))
                    temp_action = actor_container[agent].actor.sample(next_state_batch, greedy = True)
                    a_next.append(temp_action.unsqueeze(-1))
            a_next = torch.cat(a_next, -1)

            one_hot_a_next = F.one_hot(a_next, self.action_space.n)
            #print('one hot shape', one_hot_a_next.shape)
            one_hot_a_next = one_hot_a_next.reshape(batch_size, -1)
            #print('one hot shape2', one_hot_a_next.shape)
            q_next = self.critic_target(next_state_batch, one_hot_a_next)     #[batch, 1]
            target_q = reward_batch + self.gamma * mask_batch * q_next

        current_q = self.critic(state_batch, one_hot_action)
        critic_loss = (current_q - target_q).pow(2).mean()

        #actor loss
        #action_batch[:, actor_idx] = self.actor.gumbel_max_action(state_batch)
        u = F.one_hot(action_batch, self.action_space.n).float()   #need float to enable gradient
        u[:, actor_idx, :] = self.actor.gumbel_max_action(state_batch)   #[batch, num_agents, num_action]

        u = u.reshape(batch_size, -1)
        #print('state_batch shaoe', state_batch.shape)
        #print('u shape', u.shape)
        actor_loss = - self.critic(state_batch, u).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if updates % self.target_replace_iter == 0:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        return critic_loss.item(), actor_loss.item()
