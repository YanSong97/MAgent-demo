# MAgent-demo

a few attempts on env [MAgent](https://github.com/PettingZoo-Team/MAgent) 

DQN model taken from repo: [Jidi AI](https://github.com/jidiai/ai_lib/tree/master/examples/algo/dqn)

## PettingZero-MAgent wrapper

a user-friendly wrapper for enviroment *pursuit, battle, battlefield, gather, tiger_deer*. [Repo LINK](https://github.com/PettingZoo-Team/PettingZoo/tree/master/pettingzoo/magent)

**Requirement**: pettingzoo base library, simply using `pip install pettingzoo `, and also magent, use `pip install magent`


a basic demo can be found in [LINK](https://github.com/YanSong97/MAgent-demo/blob/main/Demo%20of%20PettingZero-MAgent.py)

Some key arguements for each env: [LINK](https://www.pettingzoo.ml/magent)

## Actions

<img src = 'https://github.com/YanSong97/MAgent-demo/blob/main/plot/action1.png' width  = 400>

action space: Discrete(21) as the action for 'turning' is disabled by default.

## Observation

The observation has shape [view_size x view_size x 5] by default, the last dimension consist of: 0--wall position; 1--group one position; 2--group one HP; 3---group two position; 4--group two HP. At global state, group one refers to red and group two to blue, whereas from each agent perspective, group one refers to agents in its own team and group two refers to its opponents.

## baseline models

* Independent-Q learning  (IL): individual DQN for each single agent
* Parameter-sharing Q learning  (PS): one DQN for agents in the same team
* MADDPG: Hard to train, especially in discrete action case; the number of agents varies during rollout which can also cause troubles

### Comparing success rate: (success when either team is eliminated)

20x20 battle game, 12 vs. 12, maximum episode length = 500, initial position fixed (one can also randomly shuffle the init position of agents in the same group).

<img src = 'https://github.com/YanSong97/MAgent-demo/blob/main/plot/battle-20x20%20comparison.png' width = 500>

**Brown:** parameter-sharing DQN, single-handle(ony model one agent group and the other act randomly); **Blue:** Independent-DQN, single-handle; **Green:** Independent-Q, double-handle(model both agent groups); **Red:** parameter-sharing DQN, double-handle 

**Computational consideration:** (On a macbook pro CPU) Single-handle models: <0.5h; double-handle models: around 2h.



### Render(double-handle):

IL: <img src = 'https://github.com/YanSong97/MAgent-demo/blob/main/plot/double-20x20-IL.gif' width = 400>  PS: <img src = https://github.com/YanSong97/MAgent-demo/blob/main/plot/double-20x20-PSdqn.gif width = 400>

We can rougly see a sign of ‘besige’ and 'escape'. 

## Against fixed policy opponent

A more complext env, for example, when blue team follows some attack rules: if opponents are within attackable range, then attack,  otherwise act randomly, it put red team agents initially at a disadvantage as the blue team agents will dominate the battle during early stage of training.


<img src = 'https://github.com/YanSong97/MAgent-demo/blob/main/plot/fixed%20policy.png' width = 600>  

**red:** IL, **grey:** PS; 3v3 battle game.

Compared to completely randomised oppoent policy, such advanced policy requires larger amount of computational cost for red team to learn to combat, for example as the figure illustrate, the amount of episodes goes up to >10k whereas for completely random policy only 160 is needed. training time: 6h on CPU.



## Papers on MAgent

* [Analysis of Emergent Behavior in Multi Agent Environments
using Deep Reinforcement Learning](https://ashwinipokle.github.io/assets/docs/234_final_report.pdf) (DQN, DDQN, )
* [Scalable Centralized Deep Multi-Agent
Reinforcement Learning via Policy Gradients](https://arxiv.org/pdf/1805.08776.pdf) (policy gradient)
* [Mean Field Multi-Agent Reinforcement Learning](http://proceedings.mlr.press/v80/yang18d/yang18d.pdf) (mean-field multi-agent RL)
* [Multi Type Mean Field Reinforcement Learning](https://arxiv.org/pdf/2002.02513.pdf) (mean-field)
* [Factorized Q-Learning for Large-Scale Multi-Agent Systems](https://arxiv.org/pdf/1809.03738.pdf) (factorised Q-learning)
* [PettingZoo: Gym for Multi-Agent Reinforcement
Learning](https://arxiv.org/pdf/2009.14471.pdf)


### An interactive MAgent battle games: [LINK](https://github.com/PettingZoo-Team/MAgent/blob/master/examples/show_battle_game.py)

Allows you to dispatch your solders at your command.


