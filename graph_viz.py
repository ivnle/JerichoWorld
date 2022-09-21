#%%

import random
import networkx as nx
import matplotlib.pyplot as plt

graph = nx.DiGraph()
num_plots = 10
for node_number in range(num_plots):
    # graph.add_node(node_number, Position=(random.randrange(0, 100), random.randrange(0, 100)))
    graph.add_node(node_number)
    graph.add_edge(node_number, random.choice(list(graph.nodes())))
    # nx.draw(graph, pos=nx.get_node_attributes(graph,'Position'))
    nx.draw(graph)
    plt.pause(0.5)

#%%
import jsonlines
import networkx as nx
import matplotlib.pyplot as plt

graph = nx.DiGraph()


node_set = set()
i = 0

# rom = 'new_data/dragon.jsonl'
# rom = 'new_data/zork1.jsonl'
# rom = 'new_data/ztuu.jsonl'
rom = 'new_data/sorcerer.jsonl'

with jsonlines.open(rom) as reader:
    for i, obj in enumerate(reader):

        print('ACTION:', obj['action'])
        nodes_to_add = obj['graph_add']
        nodes_to_del = obj['graph_del']

        # print(nodes_to_add)
        # print(nodes_to_del)
        
        for node in nodes_to_add:
            for j in [0, 2]:
                if node[j] not in node_set:
                    graph.add_node(node[j])
                    node_set.add(node[j])
            
            graph.add_edge(node[0], node[2], label=node[1])

        i += 1
        if i > 5:
            break
        for node in nodes_to_del:
            try:
                graph.remove_edge(node[0], node[2])
            except:
                continue
                
        pos = nx.spring_layout(graph, seed=3113794652) 
        # pos = nx.spectral_layout(graph) 
        # pos = nx.spiral_layout(graph) 
        pos = nx.planar_layout(graph) 
        nx.draw(graph, pos, node_color='lightblue')
        labels = {node: node for node in graph.nodes()}

        
        # create dict of edge tuples to edge labels
        edge_labels = {(u, v): d['label'] for u, v, d in graph.edges(data=True)}

        n = nx.draw_networkx_labels(graph, pos, labels, font_size=10, font_color="black")
        n = nx.draw_networkx_edge_labels(graph, pos, edge_labels, label_pos=0.6)

        plt.pause(3)


    
    # foo
    
        



#%%
import pandas as pd

train = pd.read_json('/home/ivanlee/JerichoWorld/data/test.json', orient='records')
train.head()

#%%
zork2 = train.iloc[3]
zork2 = zork2[zork2.notnull()]
zork2
#%%
print(zork2[2]['state']['graph'])
print(zork2[2]['next_state']['graph'])

#%%
i = 5
csg = zork2[i]['state']['graph']
print(csg)
print("action: ", zork2[i]['action'])
nsg = zork2[i]['next_state']['graph']
print(nsg)
# gd = zork2[i]['graph_diff']

nsg = ['@'.join(x) for x in nsg]
csg = ['@'.join(x) for x in csg]
added_sro = set(nsg) - set(csg)
added_sro = [x.split('@') for x in added_sro]
print('added: ', added_sro)
removed_sro = set(csg) - set(nsg)
removed_sro = [x.split('@') for x in removed_sro]
print('removed: ', removed_sro)

#%%
csg = zork2[i]
print(csg)

#%%
zork2[5]

#%%
print(zork2[1]['state']['graph'])
print(zork2[1]['next_state']['graph'])
#%%
zork2[1]

#%%
loose = train.iloc[0]

#%%
# remove None from loose
loose = loose[loose.notnull()]
loose

#%%
train.shape # 27 games

#%%
loose[0].keys()
#%%
loose[2]['graph_diff']

#%%
loose[0]['state']['graph']

#%%
loose[0]['next_state']['graph']

#%%
g1 = loose[0]['next_state']['graph']
g2 = loose[1]['state']['graph']
g1 == g2

#%%
#%%
loose[3]['state']['graph']

#%%
#%%
loose[3]['action']

#%%
g = loose[2]['state']['graph']
h = loose[3]['state']['graph']
g == h