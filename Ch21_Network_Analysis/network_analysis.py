"""
In this module, we explore graph networks. I will use the data provided in
the chapter but I have decided to write my own implementation because the
implementation of breadth first search was really poorly done. I am using
networkx package to make some nice plots of the analyzed networks.
Specifically we will try to identify the centrality or importance of certain
nodes in our network.
"""

# Perform plotting imports and networkx to create a network
from matplotlib import pyplot as plt
from collections import deque
import networkx as nx

# Define a little network of users with an id, a name and a node position
# for plotting
users = [   { "id": 0, "name": "Hero",  'pos': (0,0) },
            { "id": 1, "name": "Dunn",  'pos': (1,-1) },
            { "id": 2, "name": "Sue",   'pos': (1,1) },
            { "id": 3, "name": "Chi",   'pos': (2,0) },
            { "id": 4, "name": "Thor",  'pos': (3,0) },
            { "id": 5, "name": "Clive", 'pos': (4,0) },
            { "id": 6, "name": "Hicks", 'pos': (5,1) },
            { "id": 7, "name": "Devin", 'pos': (5,-1) },
            { "id": 8, "name": "Kate",  'pos': (6,0) },
            { "id": 9, "name": "Klein", 'pos': (7,0)  }]

# define the edges between the users as friendships 
friendships = [(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),
               (6,8),(7,8),(8,9)]

# add the friends id to each user
for user in users:
    user["friends"] =[]

for i,j in friendships:
    users[i]["friends"].append(users[j]["id"])
    users[j]["friends"].append(users[i]["id"])

# Create a graph object to plot to
G = nx.Graph()
# For each user add them to the graph object with a 'position' attribute
for user in users:
    G.add_node(user["id"],pos=user['pos'])

fig = plt.figure(1)
fig.suptitle("User-Friends Network",fontsize=14, fontweight='bold')
# get the nodes and attributes dictionary
positions = nx.get_node_attributes(G,'pos')
# add the friendships as edges to the graph
G.add_edges_from(friendships)
# draw the graph placing the nodes at the positions
nx.draw(G,positions)

# Breadth First Search Algorithm #
##################################
# We want to determine the shortest path(s) between any two nodes in our
# network and then count how many paths pass through each node. This is
# "betweeness centrality" metric.

def bfs_paths(graph, start_node, end_node):
    """ computes all the paths between the start_node and end_node in the
    users graph  using breath first search algorithm """
    # create a queue and initialize it to start at the start node
    queue = [(start_node, [start_node])]
    # keep a list of the nodes we have visited
    visited = []
    
    while queue:
        # get the first node, path pair
        (node, path) = queue.pop(0)
        # get the adjacent nodes (in this case friends)
        for next in graph[node]["friends"]:
            # make sure we haven't yet visited these nodes
            if next not in visited:
                # if one of the adjacents is our end_node then we yield it.
                # Note this will yield the shortest path first but there may
                # be multiple shortest paths
                if next == end_node:
                    yield path + [next]
                else:
                    # otherwise add the adjacent node and appended path to
                    # the queue
                    queue.append((next, path + [next]))
            # update the nodes we have visited
            visited.append(node)

def shortest_paths(graph, start_node, end_node):
    """ returns the shortest paths bewtween start and end_nodes in graph """
    # get all the paths between the start and end node
    paths = list(bfs_paths(graph, start_node, end_node))
    # calculate the shortest length
    min_path_length = min([len(row) for row in paths])
    # filter for only shortest paths
    shortest_paths = [path for path in paths if len(path) <=min_path_length]
    return shortest_paths

# Calulate all Shortest Paths #
###############################
# now we are ready to calculate for each node in our network the shortest
# paths to all other nodes in the network

# add shortest path and betweeness centrality keys to users
for user in users:
    user["shortest_paths"] = []
    user['betweeness_centrality'] = 0.0

# for each start and end node call shortest_paths to get all the shortest
# paths and append to users
for start_node, _ in enumerate(users):
    for end_node, _ in enumerate(users):
        if start_node != end_node:
            users[start_node]['shortest_paths'].append(shortest_paths(users,
            start_node, end_node))

# Betweeness Centrality #
#########################
# for each pair start/end nodes 
for start_node,_ in enumerate(users):
    for end_node, _ in enumerate(users):
        if start_node < end_node:
            # get all the paths, remember the nodes start at 0
            paths = users[start_node]['shortest_paths'][end_node-1]
            # get the number of paths
            num_paths = len(paths)
            # the contibution for each id in the path should be 1/n paths
            contribution = 1/float(num_paths)
            # add the contribution to each id in each path
            for path in paths:
                for id in path:
                    if id not in [start_node, end_node]:
                        users[id]['betweeness_centrality'] += contribution

# Let the size of each node be the centrality metric, 50 is arbitrary, size
# is normally 300, what matters here is size difference between nodes
sizes = [50*user['betweeness_centrality'] for user in users]

fig = plt.figure(2)
fig.suptitle("Betweeness-Centrality of User-Friends Network",fontsize=14, 
              fontweight='bold')
# get the nodes and attributes dictionary
positions = nx.get_node_attributes(G,'pos')
# add the friendships as edges to the graph
G.add_edges_from(friendships)
# draw the graph placing the nodes at the positions
nx.draw(G,positions,node_size=sizes)
plt.show()
