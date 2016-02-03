# This script reads the addresses_csv file (a list of addresses). The file has comma separated fields and header information. We save the data to a dictionary and then play around with saving the dictionary for later use

import sys
import csv
import  pickle
from collections import defaultdict

try:
    # obtain the file_name from stdin
    file_name = sys.argv[1]
except:
    print "usage: read_csv.py file_name"
    sys.exit(1) # non-zero exit code to indicate error

# create a defaultdict for accumulating addresses
addresses = defaultdict(list)

with open(file_name, 'rb') as f:
    # call the csv dictionary reader method and assign key value pairs
    reader = csv.DictReader(f, delimiter = ',')
    for row in reader:
        addresses['UniqueID'].append(int(row["UNIQUEID"]))
        addresses['StreetAddress'].append(row["StreetAddress"])
        addresses['City'].append(row["City"])
        addresses['State'].append(row["State"])
        addresses['Zip'].append(int(row["Zip"]))

# Create an output file to save the addresses dict to 
output = open('addresses.pkl','wb')
pickle.dump(addresses,output)
output.close()
print "pickle saved"

# ask the user if they want to re-open
open_pickle = raw_input("Re-open pickle?")
if open_pickle in ('yes', 'ye', 'y'):
    pkl_file = open('addresses.pkl','rb')
    data = pickle.load(pkl_file)
    # print to stdout
    print data
else: print "pickle not opened"


