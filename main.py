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
    examine: dict = None
    valid_actions: list = None
    graph: list = None
    
    

def load_attributes():
    global attributes
    global readable
    global MOVE_ACTIONS
    MOVE_ACTIONS = 'north/south/west/east/northwest/southwest/northeast/southeast/up/down/enter/exit'.split('/')

    with open('symtables/readable_tables.txt', 'r') as f:
        readable = [str(a).strip() for a in f]

    attributes = {}
    attr_vocab = set()
    for gn in readable:
        attributes[gn] = {}

        with open('symtables/' + gn + '.out', 'r') as f:
            try:
                for line in f:
                    if "attribute" in line.lower():
                        split = line.split('\t')
                        if len(split) < 2:
                            split = line.split()
                            idx, attr = int(split[1]), split[2]
                            if len(split) < 2:
                                continue
                        else:
                            idx, attr = int(split[0].split(' ')[1]), split[1]
                        if '/' in attr:
                            attr = attr.split('/')[0]
                        attributes[gn][idx] = attr.strip()
                        attr_vocab.add(attr.strip())

            except UnicodeDecodeError:
                print("Decode error:", gn)
                continue

def build_kg(env, prev_act=None, prev_loc=None):
    triples = set()

    # Get player's location
    loc = env.get_player_location()

    # Get player object
    player = env.get_player_object()
    
    # Get location object
    loc = env.get_object(player.parent)
    loc_name = loc.num if not loc.name else loc.name
    triples.add(('you', 'in', loc_name))

    # Get objects in location
    player_subtree = jericho.util.get_subtree(player.num, env.get_world_objects())

    # Get objects in current location (but not in inventory)
    for obj in player_subtree:        
        obj_name = obj.num if not obj.name else obj.name
        if obj.num == player.num:
          continue
        elif obj.parent == loc.num:          
          triples.add((obj_name, 'in', loc_name))
        elif obj.parent == player.num:
          triples.add(('you', 'has', obj_name))
        else:
          obj_parent = env.get_object(obj.parent)
          obj_parent_name = obj_parent.num if not obj_parent.name else obj_parent.name
          triples.add((obj_name, 'in', obj_parent_name))

    # Current location relative to previous location
    # if prev_loc:
    #   if prev_act.lower() in MOVE_ACTIONS:
    #       triples.add((loc.name, prev_act.replace(' ', '_'), prev_loc.name))

    #   if prev_act.lower() in jericho.defines.ABBRV_DICT.keys():
    #       prev_act = jericho.defines.ABBRV_DICT[prev_act.lower()]
    #       triples.add((loc.name, prev_act.replace(' ', '_'), prev_loc.name))

    return triples

def clean(text):
  bad_text = ["A strange little man in a long cloak appears suddenly in the room. He is wearing a high pointed hat embroidered with astrological"]
  for bt in bad_text:
    if bt in text:
      # remove bt from text and all the text after
      head, sep, tail = text.partition(bt)
      text = head
  return text


def examine_objects(env, state):
  obj_desc = {}

  # Get player object
  player = env.get_player_object()

  # Get location object
  loc = env.get_object(player.parent)

  # Get objects in location
  player_subtree = jericho.util.get_subtree(player.num, env.get_world_objects())

  for obj in player_subtree:
    if obj.num != player.num:
      desc = jericho.util.clean(env.step('examine ' + obj.name)[0])
      env.set_state(state)    
      
      desc = clean(desc)

      obj_desc[obj.name] = desc

  return obj_desc


def update_graph(graph, next_graph, env, act):
  
  # make copy of graph

  graph_copy = graph.copy()
  triples_to_add = next_graph - graph_copy

  loc_name = [t for t in graph_copy if (t[0] == 'you' and t[1]=='in')][0]
  loc_name = loc_name[2]

  for triple in triples_to_add:
    sub, rel, obj = triple
  
    # Move from surrounding object to inventory
    if sub == 'you' and rel == 'has':
      print(loc_name)
      graph_copy.discard((obj, 'in', loc_name))
    
    if sub == 'you' and rel == 'in':
      graph_copy.discard(('you', 'in', loc_name))

      if act.lower() in MOVE_ACTIONS:
          graph_copy.add((obj, act.replace(' ', '_'), loc_name))

      if act.lower() in jericho.defines.ABBRV_DICT.keys():
          act = jericho.defines.ABBRV_DICT[act.lower()]
          graph_copy.add((obj, act.replace(' ', '_'), loc_name))
      

    graph_copy.add(triple)
  
  return graph_copy
  

def build_dataset(rom):
    env = jericho.FrotzEnv(rom)
    walk = env.get_walkthrough()
    data = []
    action = ''
    next_graph = None
    # prev_act = None
    prev_loc = None

    # for act in walk:
    for i, act in enumerate(walk[:50]):
        state = env.get_state()

        # Record state before step
        look = clean(env.step('look')[0])
        env.set_state(state)

        obs = '' if i == 0 else next_obs

        inv_desc = clean(env.step('inventory')[0])
        env.set_state(state)
        graph = next_graph if next_graph else build_kg(env)

        examine = examine_objects(env, state)

        state_text = State(obs=obs, look=look,
                           inventory=inv_desc, graph=list(graph),
                           valid_actions=env.get_valid_actions(), examine=examine)

        prev_loc = env.get_player_location()

        # Take a step and save state
        next_obs, rew, done, info = env.step(act)
        state = env.get_state()

        next_look = clean(env.step('look')[0])
        env.set_state(state)
        next_inv_desc = clean(env.step('inventory')[0])
        env.set_state(state)
        next_graph = build_kg(env, act, prev_loc)
        next_examine = examine_objects(env, state)

        # next_graph = update_graph(graph, next_graph, env, act)

        next_state_text = State(obs=next_obs, look=next_look,
                                inventory=next_inv_desc, graph=list(
                                    next_graph),
                                valid_actions=env.get_valid_actions(), examine=next_examine)

        
        

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
    
    # roms = ["roms/snacktime.z8"]
    
    # Fantasy
    # roms = ["roms/zork1.z5"]
    # roms = ["roms/zork2.z5"]
    roms = ["roms/zork3.z5"]
    # roms = ["roms/dragon.z5"]
    # roms = ["roms/deephome.z5"]

    load_attributes()

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
        you = env.get_player_objec
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
