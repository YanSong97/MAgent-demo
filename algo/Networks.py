import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

def weights_init_kaiming(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def weights_init_xavier(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


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



################################ MADDPG  ####################################################


class Actor(nn.Module):
    """
    we relax the discrete action space using Gumbel-softmax
    """

    def __init__(self, action_dim, in_size=128, hidden_size=128):

        super(Actor, self).__init__()

        self.conv = CNN_encoder()

        self.linear1 = nn.Linear(in_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size,hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_dim)  # logits
        self.apply(weights_init_xavier)

    def forward(self, state):
        """
        [batch, state_dim]
        """
        conv_encoded = self.conv(state)

        x = F.relu(self.linear1(conv_encoded))
        x = F.relu(self.linear2(x))
        logits = self.linear3(x)
        return logits

    def gumbel_max_action(self, state):
        logit = self.forward(state)
        return F.gumbel_softmax(logit, hard = True)

    def sample(self, state, greedy):
        logit = self.forward(state)
        if greedy:
            prob = F.softmax(logit, -1)
            return prob.argmax(-1)
        else:
            dist = Categorical(logits = logit)
            return dist.sample()

    def prob(self, state):
        logit = self.forward(state)
        return F.softmax(logit, -1)






class centralised_critic(nn.Module):
    def __init__(self, num_agents, action_size,input_size = 128, hidden_size = 128):
        super(centralised_critic, self).__init__()

        self.num_agents = num_agents
        self.action_size = action_size

        self.conv = CNN_encoder()
        #print('input_size = ', input_size)
        #print('num agents = ', num_agents)
        #print('action size = ', action_size)
        self.linear1 = nn.Linear(input_size + int(num_agents*action_size), hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.apply(weights_init_kaiming)

    def forward(self, view_state, agent_actions):
        """
        :param view_state:  [batch, width, height, num_channels]
        :param agent_actions:  [batch, num_agents*action_size]
        :return:
        """
        #print('view space shape', view_state.shape)
        #print('agent action shape', agent_actions.shape)

        conv_encoded = self.conv(view_state)
        state_action = torch.cat([conv_encoded, agent_actions], dim = -1)
        s_a = F.relu(self.linear1(state_action))
        s_a = F.relu(self.linear2(s_a))
        q = self.linear3(s_a)

        return q



