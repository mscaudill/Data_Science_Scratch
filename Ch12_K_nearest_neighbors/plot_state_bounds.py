""" This script plots the state boundaries specified in the text file
state_boundaries.txt """

import re
from collections import defaultdict

# Open the text file for reading #
##################################
state_borders = defaultdict(list)
with open('state_boundaries.txt','rb') as f:
    current_state = ''
    
    for line in f:

        if re.search('state name', line):
            # get the state name
            state_name = re.search(r'(state name=")(\w+)',line).group(2)
            state_borders[state_name]
            current_state = state_name

        elif re.search('point', line):
            # get the latitude and longitude of this line
            latitude = float(re.search(r'(lat=")([\d.-]+)', line).group(2))
            longitude = float(re.search(r'(lng=")([\d.-]+)',line).group(2))
            state_borders[current_state].append([latitude,longitude])
            
print state_borders
