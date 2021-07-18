# MAgent-demo

a few attempts on env [MAgent](https://github.com/PettingZoo-Team/MAgent) 

DQN model taken from repo: [Jidi AI](https://github.com/jidiai/ai_lib/tree/master/examples/algo/dqn)

## PettingZero-MAgent wrapper

a user-friendly wrapper for enviroment *pursuit, battle, battlefield, gather, tiger_deer*. [Repo LINK](https://github.com/PettingZoo-Team/PettingZoo/tree/master/pettingzoo/magent)

**Requirement**: pettingzoo base library, simply using `pip install pettingzoo `, and also magent, use `pip install magent`


a basic demo can be found in [LINK](https://github.com/YanSong97/MAgent-demo/blob/main/Demo%20of%20PettingZero-MAgent.py)

Some key arguements for each env: [LINK](https://www.pettingzoo.ml/magent)



## baseline models

* Independent-Q learning
* Parameter-sharing Q learning

### Compare success rate:

<img src = 'https://github.com/YanSong97/MAgent-demo/blob/main/plot/battle-20x20%20comparison.png' width = 500>

**Brown:** parameter-sharing DQN, single-handle(ony model one agent group and the other act randomly); **Blue:** Independent-DQN, single-handle; **Green:** Independent-Q, double-handle(model both agent groups); **Red:** parameter-sharing DQN, double-handle 


### Render:

IL: <img src = 'https://github.com/YanSong97/MAgent-demo/blob/main/plot/double-20x20-IL.gif' width = 400>      PS-Q: <img src = https://github.com/YanSong97/MAgent-demo/blob/main/plot/double-20x20-PSdqn.gif width = 400>




This figure shows the success rate (success when either team is eliminated) during testing phase, on a 20x20 battle env (episode length=500) with 12 agents on each side. The computational time for the single-handle is < 0.5h and around 2 hours for double-handle, on a macbook cpu.



Animation of two consequtive battle games during training between red and blue armies with 12 agents each, using IL on both agent groups.  We can rougly see a sign of ‘besige’  'escape'. 



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


