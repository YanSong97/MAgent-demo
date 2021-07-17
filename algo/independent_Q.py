
from torch.optim import Adam

from .utils import *
from .Networks import *


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








