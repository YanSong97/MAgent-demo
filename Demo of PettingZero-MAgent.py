# requirement: pettingzoo library, use pip install pettingzoo


from pettingzoo.magent import adversarial_pursuit_v3
from pettingzoo.magent import battle_v3
from pettingzoo.magent import battlefield_v3
from pettingzoo.magent import gather_v3
from pettingzoo.magent import tiger_deer_v3

env_name = 'tiger_deer'
if_render = True
map_size = 50

if env_name == 'pursuit':
    env = adversarial_pursuit_v3.parallel_env(map_size = map_size)
elif env_name == 'battle':
    env = battle_v3.parallel_env(map_size=map_size)
elif env_name == 'battlefield':
    env = battlefield_v3.parallel_env(map_size = map_size)
elif env_name == 'gather':
    env = gather_v3.parallel_env()    #to tune the map_size we need to modify the gather_v3.py file, particurly the input of parallel_env fn
elif env_name == 'tiger_deer':
    env = tiger_deer_v3.parallel_env(map_size = map_size)
else:
    raise NotImplementedError


done = False
state = env.reset()

key_list = env.possible_agents
action_space = env.action_spaces

while not done:
    action_dict = {key_list[i]: action_space[key_list[i]].sample() for i in range(len(key_list))}

    next_state, reward_dict, done_dict, _ = env.step(action_dict)

    done = all(done_dict.values())
    if if_render:
        env.render()


