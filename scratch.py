#%%
from argparse import Action
from re import L
import jsonlines
import pandas as pd
import difflib
#%%
# rom = 'new_data/ztuu.jsonl'
# rom = 'new_data/zork1.jsonl'
# rom = 'new_data/karn.jsonl'
rom = 'new_data/dragon.jsonl'
# rom = 'new_data/sorcerer.jsonl'
# rom = 'new_data/planetfall.jsonl'

# read rom as json lines as dataframe
df = pd.read_json(rom, lines=True)
df.head()

#%%
# get first row of df
print(df.iloc[0]['state']['obs'])
print("-"*80)
print(df.iloc[0]['next_state']['obs'])

#%%
def get_location(graph):
    for s, r, o in graph:
        if s == 'you' and r == 'in':
            return o
    return None

i = 0
g_0 = df.iloc[i]['state']['graph']
g_1 = df.iloc[i]['next_state']['graph']
loc_0 = get_location(g_0)
loc_1 = get_location(g_1)
print(loc_0, loc_1)

#%%
# filter rows where location doesn't change
df = df[df.apply(lambda x: get_location(x['state']['graph']) == get_location(x['next_state']['graph']), axis=1)]
df.head()

#%%
# get state and next_state values for 'look'
df = df[df.apply(lambda x: x['state']['look'] != x['next_state']['look'], axis=1)]
df.head()
print(len(df))

#%%
# print both state's looks
for i in range(10):
    action = df.iloc[i]['action']
    look_0 = df.iloc[i]['state']['look']    
    look_1 = df.iloc[i]['next_state']['look']
    print(look_0)
    print("-"*80)
    print(look_1)
    print("-"*80)
    # print differences between look_0 and look_1
    print('action: ', action)
    for line in difflib.unified_diff(look_0.splitlines(), look_1.splitlines(), fromfile='look_0', tofile='look_1'):
        print(line)
    print("="*80)

