"""

"""

# Perform plotting imports and networkx to create a network
from matplotlib import pyplot as plt
import networkx as nx

users = [   { "id": 0, "name": "Hero", 'pos': (0,0) },
            { "id": 1, "name": "Dunn", 'pos': (1,-1) },
            { "id": 2, "name": "Sue", 'pos': (1,1) },
            { "id": 3, "name": "Chi", 'pos': (2,0) },
            { "id": 4, "name": "Thor", 'pos': (3,0) },
            { "id": 5, "name": "Clive", 'pos': (4,0) },
            { "id": 6, "name": "Hicks", 'pos': (5,1) },
            { "id": 7, "name": "Devin", 'pos': (5,-1) },
            { "id": 8, "name": "Kate", 'pos': (6,0) },
            { "id": 9, "name": "Klein", 'pos': (7,0)  }]

friendships = [(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),
               (6,8),(7,8),(8,9)]

G = nx.Graph()
# For each user add them to the graph object with a 'position' attribute
for user in users:
    G.add_node(user["id"],pos=user['pos'])

# get the nodes and attributes dictionary
positions = nx.get_node_attributes(G,'pos')
# add the friendships as edges to the graph
G.add_edges_from(friendships)
# draw the graph placing the nodes at the positions
nx.draw(G,positions)

plt.show()
