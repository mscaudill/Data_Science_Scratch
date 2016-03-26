"""

"""

# Perform plotting imports and networkx to create a network
from matplotlib import pyplot as plt
from collections import deque
import networkx as nx


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

friendships = [(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),
               (6,8),(7,8),(8,9)]

for user in users:
    user["friends"] =[]

for i,j in friendships:
    users[i]["friends"].append(users[j]['id'])
    users[j]["friends"].append(users[i]['id'])


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
plt.show()i

def shortest_paths_from(from_user):
    
    # a path will be represented as a list of users
    
    # a dictionary from "user_id" to *all* shortest paths to that user, if
    # there are multiple shortest paths it will contain them all
    shortest_paths_to = {from_user["id"]: [[]]}

    # a queue of (previous user, next user) that we need to explore. It will
    # start out with all pairs (from_user, friend_from_user). Think of these
    # as little path segements.
    frontier = deque((from_user, friend) 
                      for friend in from_user["friends"])

    # Continue until we exhaust the queue
    while edges:
        
        # get and remove the first user in the queue
        prev_user, user = edges.popleft()
        user_id = user["id"]

        # the shortest path from the user to the previous user is just the
        # previous user id path. Note there might be several previous 
        # users so we add them all to new_paths_to_user
        paths_to_prev_user = shortest_paths_to[prev_user["id"]])
        new_paths_to_user = [path + [user_id] 
                                for path in paths_to_prev_user]

        # it is possible that we already have the shortest path
        old_paths_to_user = shortest_paths_to.get(user_id,[])

        # what is the shortest path to this user we have seen so far
        if old_paths_to_user:
            min_path_length = len(old_paths_to_user[0])
        else:
            min_path_length = float('inf')

        # only keep paths that aren't too long and are new
        new_paths_to_user = [path for path in new_paths_to_user
                             if len(path) <= min_path_length
                             and path not in old_paths_to_user]
        
        # This handles the case of multiple shortest paths to the user by
        # adding the new path in 
        shortest_paths_to[user_id] = old_paths_to_user + new_paths_to_user

        # add never-before-seen neighbors to the frontier
        edges.extend((user, friend) for friend in user['friends']
                         if friend not in shortest_paths_to)

        return shortest_paths_to
