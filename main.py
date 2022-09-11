# %%
import pandas as pd
import jericho
from glob import glob
import json
import jsbeautifier
from dataclasses import dataclass, asdict


@dataclass
class State:
    obs: str = ''
    look: str = ''
    inventory: str = ''
    graph: list = None
    valid_actions: list = None


def build_kg(env, look):
    triples = set()

    # Get player's location
    loc = env.get_player_location()

    if not loc.name:
        print('No location name!')
        loc_name = look.split('\n')[0]
        loc_name = jericho.util.clean(loc_name)
    else:
        loc_name = loc.name

    triples.add(('you', 'in', loc_name))

    # print(jericho.defines.ABBRV_DICT)

    # Get objects in location
    subtree = jericho.util.get_subtree(loc.num, env.get_world_objects())

    o_nb = {}
    for obj in subtree:
        if obj.num == env.get_player_object().num:
            continue
        elif obj.parent == loc.num:
            if not obj.name:
                triples.add((obj.num, 'in', loc_name))
                o_nb[obj.num] = obj.num
            else:
                triples.add((obj.name, 'in', loc_name))
                o_nb[obj.num] = obj.name

    for obj in subtree:
        if obj.parent in o_nb:
            if not obj.name:
                triples.add((obj.num, 'in', o_nb[obj.parent]))
            else:
                triples.add((obj.name, 'in', o_nb[obj.parent]))

    # Get inventory
    inv = env.get_inventory()
    for item in inv:
        triples.add(('you', 'has', item.name))

    return triples


def build_dataset(rom):
    env = jericho.FrotzEnv(rom)
    walk = env.get_walkthrough()
    data = []
    action = ''

    # for act in walk:
    for i, act in enumerate(walk[:5]):
        state = env.get_state()

        # Record state before step
        look = env.step('look')[0]
        env.set_state(state)

        obs = '' if i == 0 else next_obs

        inv_desc = env.step('inventory')[0]
        env.set_state(state)
        graph = build_kg(env, obs)

        state_text = State(obs=obs, look=look,
                           inventory=inv_desc, graph=list(graph),
                           valid_actions=env.get_valid_actions())

        # Take a step and save state
        next_obs, rew, done, info = env.step(act)
        state = env.get_state()


        next_look = env.step('look')[0]
        env.set_state(state)
        next_inv_desc = env.step('inventory')[0]
        env.set_state(state)
        next_graph = build_kg(env, next_look)
        next_state_text = State(obs=next_obs, look=next_look,
                                inventory=next_inv_desc, graph=list(next_graph),
                                valid_actions=env.get_valid_actions())

        sample = {
            'state': asdict(state_text),
            'action': act,
            'next_state': asdict(next_state_text),
            'graph_add': list(next_graph - graph),
            'graph_del': list(graph - next_graph),
            'reward': rew,
        }
        data.append(sample)
        options = jsbeautifier.default_options()
        options.indent_size = 2
        print(jsbeautifier.beautify(json.dumps(sample), options))
        print('----------------------------------------')
        # print(sample, '\n')


if __name__ == '__main__':

    # roms = glob("roms/*.z*")
    # roms = ["roms/zork1.z5"]
    # roms = ["roms/snacktime.z8"]
    roms = ["roms/zork1.z5"]
    for rom in roms:
        build_dataset(rom)


foo
# Create the environment, optionally specifying a random seed
# rom = "roms/snacktime.z8"
# rom = "roms/zork1.z5"

for rom in roms[2:3]:
    print('----------------------------------------')
    print('rom:', rom)

    env.seed()
    obs = env.reset()
    print(obs)

    for act in walkthru[:10]:
        print('----------------------------------------')
        # print(env.get_state())
        loc = env.get_player_location()
        you = env.get_player_object()
        inv = env.get_inventory()
        print('inventory:', inv)

        # print('you_obj:', you)
        obs = env.step('look')[0]
        print('obs:', obs)
        actions = env.get_valid_actions()
        print(actions)
        # print(obs.split('\n')[0])

        if not loc.name:
            print('No location name!')
            loc_name = obs.split('\n')[0]
            loc_name = jericho.util.clean(loc_name)
        else:
            loc_name = loc.name

        print('location name:', loc_name)

        print('action:', act)
        obs = env.step(act)[0]
        print('obs2:', obs)
        inv = env.get_inventory()
        print('inventory:', inv)
        # break
    # env.reset()

# while not done:
#     # Take an action in the environment using the step fuction.
#     # The resulting text-observation, reward, and game-over indicator is returned.
#     observation, reward, done, info = env.step('open mailbox')
#     # Total score and move-count are returned in the info dictionary
#     print('Total Score', info['score'], 'Moves', info['moves'])
# print('Scored', info['score'], 'out of', env.get_max_score())

# %%

# read data/train.json
train = pd.read_json(
    '/home/ivanlee/JerichoWorld/data/train.json', orient='records')
train.head()

# %%
s = train.iloc[3, 15]
print(s.keys())
print('rom:', s['rom'])

# %%
s['state']

# %%
s['next_state']

# %%
# %%
s['state'].keys()

# %%
s['state']

# %%
s['next_state']

# %%
s['action']

'(((36, 100),), ((4, 7),), ((4, 7), (36, 14), (33, 14), (36, 14), (33, 14)))'

# %%
s['action']

'((), ((4, 7), (102, 11)), ((4, 7), (36, 14), (33, 14), (36, 14), (33, 14)))'

# %%
s.keys()
# dict_keys(['rom', 'state', 'next_state', 'graph_diff', 'action', 'reward'])

# %%
s['graph_diff']
# %%
s['state']['graph']
# %%
s['next_state']['graph']

# %%
s['action']

"""
rom:enchant
g_obs(graph_t, action_t) -> graph_t+1 (how will the world change if i perform a particular action?)
obs(graph_t, action_t) -> str: observation
look(graph) -> str: loc_desc
inventory(graph) -> str: inv_desc
actions(graph) -> list: actions (what actions can i perform?)

[['frotz ', 'in', 'book'],
  ['loaf bread', 'in', 'oven'],
  ['you', 'have', 'jug'],
  ['oven', 'in', 'Inside Shack'],
  ['you', 'have', 'battered lantern'],
  ['you', 'have', 'book'],
  ['gnus', 'in', 'book'],
  ['nitfol ', 'in', 'book'],
  ['you', 'in', 'Inside Shack'],
  ['blorb ', 'in', 'book']]

action: 'blorb book'

[['frotz ', 'in', 'book'],
  ['book', 'in', 'strong box'],
  ['loaf bread', 'in', 'oven'],
  ['you', 'have', 'jug'],
  ['strong box', 'in', 'Inside Shack'],
  ['oven', 'in', 'Inside Shack'],
  ['you', 'have', 'battered lantern'],
  ['gnus', 'in', 'book'],
  ['nitfol ', 'in', 'book'],
  ['you', 'in', 'Inside Shack'],
  ['blorb ', 'in', 'book']]


rom:zork2
[['you', 'have', 'lamp'],
  ['Narrow Tunnel', 'south', 'Inside Barrow'],
  ['you', 'in', 'Narrow Tunnel'],
  ['you', 'have', 'sword']],

action: S

[['you', 'have', 'lamp'],
  ['Foot Bridge', 'south', 'Narrow Tunnel'],
  ['you', 'in', 'Foot Bridge'],
  ['you', 'have', 'sword']],


"""

# %%
s['action']
