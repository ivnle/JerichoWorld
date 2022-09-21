# %%
import pandas as pd
import jericho
from glob import glob
import json
import jsonlines
import jsbeautifier
from dataclasses import dataclass, asdict
import tqdm


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


def get_name(obj):
  try:
    if isinstance(obj, jericho.jericho.ZObject):
      name = f'{(obj.name).strip()}_{obj.num}'
    else:
      name = f'Null'
    return name
  except:
    print('PROBLEM OBJECT:', obj)
    foo

def build_kg(env, prev_act=None, prev_loc=None):

    triples = set()

    # Get player object
    player = env.get_player_object()
    
    # Get location object
    loc = env.get_object(player.parent)    
    if loc is None:
      loc = env.get_player_location()
    if loc is None:
      loc = 'null_loc'

    triples.add((f'you', 'in', get_name(loc)))

    # Get player subtree including siblings
    player_subtree = jericho.util.get_subtree(loc.num, env.get_world_objects())
    # print(player_subtree)
    # foo
    obj_in_graph = []
    for obj in player_subtree:        
        obj_name = obj.num if not obj.name else obj.name
        # ignore player object
        if obj.num == player.num:
          continue
        # surrounding object
        elif obj.parent == loc.num:          
          triples.add((f'{obj.name}_{obj.num}', 'in', get_name(loc)))
          obj_in_graph.append(obj.num)
        # inventory object
        elif obj.parent == player.num:
          triples.add(('you', 'has', get_name(obj)))
          obj_in_graph.append(obj.num)
        # nested objects
        # else:
        #   obj_parent = env.get_object(obj.parent)
        #   obj_parent_name = obj_parent.num if not obj_parent.name else obj_parent.name
        #   triples.add((obj_name, 'in', obj_parent_name))
    for obj in player_subtree:      
      if obj.parent in obj_in_graph:
        obj_name = obj.num if not obj.name else obj.name
        obj_parent = env.get_object(obj.parent)
        obj_parent_name = obj_parent.num if not obj_parent.name else obj_parent.name
        triples.add((get_name(obj), 'in', get_name(obj_parent)))

    # Current location relative to previous location
    if prev_loc:
      if prev_act.lower() in MOVE_ACTIONS:
          triples.add((get_name(loc), prev_act.replace(' ', '_'), get_name(prev_loc)))

      if prev_act.lower() in jericho.defines.ABBRV_DICT.keys():
          prev_act = jericho.defines.ABBRV_DICT[prev_act.lower()]
          triples.add((get_name(loc), prev_act.replace(' ', '_'), get_name(prev_loc)))

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
  triples_to_del = graph_copy - next_graph

  loc_name = [t for t in graph_copy if (t[0] == 'you' and t[1]=='in')][0]
  loc_name = loc_name[2]

  # Loop through added triples
  for triple in triples_to_add:
    sub, rel, obj = triple
  
    # Move from surrounding object to inventory
    if sub == 'you' and rel == 'has':
      # print(loc_name)
      for g_trip in graph:
          s, r, o = g_trip
          if s == obj and r == 'in':
            graph_copy.discard(g_trip)
    
    # player moved to new location so update their location
    elif sub == 'you' and rel == 'in':
      graph_copy.discard(('you', 'in', loc_name))

      if act.lower() in MOVE_ACTIONS:
          graph_copy.add((obj, act.replace(' ', '_'), loc_name))

      if act.lower() in jericho.defines.ABBRV_DICT.keys():
          act = jericho.defines.ABBRV_DICT[act.lower()]
          graph_copy.add((obj, act.replace(' ', '_'), loc_name))

    graph_copy.add(triple)

  # Loop through deleted triples
  for triple in triples_to_del:
    sub, rel, obj = triple

    # Remove from inventory
    if sub == 'you' and rel == 'has':
      graph_copy.discard(('you', 'has', obj))

    # surrounding object removed from current location
    # may or may not be moved into inventory
    # elif (rel == 'in') and (obj == loc_name):
    #   graph_copy.discard((sub, 'in', obj))
  
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
    for i, act in enumerate(tqdm.tqdm(walk)):

        # Get initial observation at the start
        if i==0:
          init_obs, init_info = env.reset()
          obs = init_obs
        # Set obs to obs from previous step
        else:
          obs = next_obs
        
        # Save state before step
        state = env.get_state()        
        
        # Get location description
        look = clean(env.step('look')[0])
        env.set_state(state)

        # Get inventory description
        inv_desc = clean(env.step('inventory')[0])
        env.set_state(state)

        # Build knowledge graph from object tree
        graph = next_graph if next_graph else build_kg(env)

        # Get descriptions for each inventory object
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

        next_graph = update_graph(graph, next_graph, env, act)

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
        # options = jsbeautifier.default_options()
        # options.indent_size = 2
        # print(jsbeautifier.beautify(json.dumps(sample), options))
        # print('----------------------------------------')
        # print(sample, '\n')
    return data


if __name__ == '__main__':

    # roms = glob("roms/*.z*")
    
    # roms = ["roms/snacktime.z8"]
    
    # Fantasy
    roms = ["roms/ztuu.z5", "roms/sorcerer.z3", "roms/zork1.z5", "roms/dragon.z5", "roms/deephome.z5"]

    load_attributes()

    for rom in roms:
        data = build_dataset(rom)
        fp = rom.split('/')[-1].split('.')[0] + '.jsonl'
        with jsonlines.open(f'new_data/{fp}', 'w') as writer:
            writer.write_all(data)


