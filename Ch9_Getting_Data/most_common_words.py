"""
most_common_words.py

An example of how to read in a text file and extract some simple info

"""

import sys
from collections import Counter

try: 
    # show the top sys.argv[1] number of words in stdout 
    num_words = int(sys.argv[1])
except:
    print "usage: most_common_words.py num_words"
    sys.exit(1) # any non-zero exit code indicates error

# We first lower case all words, 
#(2) strip leading and trailing spaces and 
#(3) split on the white spaces 
#(4) count only if a word
counter = Counter(word.lower() for line in sys.stdin 
                  for word in line.strip().split() 
                  if word)
# use the most_common method of the counter class
for word,count in counter.most_common(num_words):
    sys.stdout.write(word)
    sys.stdout.write("\t")
    sys.stdout.write(str(count))
    sys.stdout.write("\n")
