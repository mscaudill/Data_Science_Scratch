""" This script plots the state boundaries specified in the text file
state_boundaries.txt """

import re
from collections import defaultdict
from matplotlib import pyplot as plt

# Open the text file for reading #
##################################
# create a dictionary to hold the latitudes and longitudes of each state
state_borders = defaultdict(list)
# open the state_boundaries text file
with open('state_boundaries.txt','rb') as f:
    # keep track of the current state we are extracting lat,long on
    current_state = ''
    
    for line in f:
        if re.search('state name', line):
            # get the state name, It may be two parts; New Mexico
            state_name = re.search(
                            r'(state name=")(\w+\s*\w+)',line).group(2)
            # add to our dict
            state_borders[state_name]
            current_state = state_name

        elif re.search('point', line):
            # get the latitude and longitude of this line, search for
            # numeric and allow . and hypens
            latitude = float(re.search(r'(lat=")([\d.-]+)', line).group(2))
            longitude = float(re.search(r'(lng=")([\d.-]+)',line).group(2))
            # add these to the current state
            state_borders[current_state].append([latitude,longitude])

# Plot longitudes and latitudes for each state #
################################################
def plot_states(plt, color = '0.8'):
    for state in state_borders:
        # get all the latitudes of this state
        latitudes = [state_borders[state][point][0] for point,_ in 
                     enumerate(state_borders[state])]
        # get all the longitudes of this state
        longitudes = [state_borders[state][point][1] for point,_ in 
                      enumerate(state_borders[state])]
        # ignore Alaska and Hawaii and plot each state
        if state not in ('Alaska', 'Hawaii'):
            plt.plot(longitudes,latitudes, color = color)



