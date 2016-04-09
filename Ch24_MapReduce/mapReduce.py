"""
MapReduce is a programming model for performing parallel processing on large
data sets. There are essentially three steps to mapReduce.
1. use a mapper func to turn each item into (key, value) pairs.
2. collect together all the pairs with identical keys into groups (dict obj)
3. use a reducer function to produce outputs for each key
"""
from DS_Scratch.Ch13_Naive_Bayes.spam_classifier import tokenize
from collections import defaultdict

""" A basic starting example with mapReduce is to count words in documents.
Below we will look at the classic way to do this (without mapReduce) and
then the distributed mapReduce approach."""

# Basic word counting #
#######################
# Lets say you wanted to count the most popular words in from a set of
# status updates on twitter etc.. So far we have seen how to do this with 
def word_count(documents):
    """ basic word counting by looping through each document and counting
    words. """
    return Counter(word for document in documents 
                   for word in tokenize(document))

""" Notice that this function can only work on one document at a time and it
needs access to all the documents. If there are billions of them this will
be very slow. MapReduce will allow us to distribute the documents across
processing clusters. Lets implement it for word counting in documents. """

# MapReduce word counting #
###########################
""" 1. write a mapper function that returns key,value pairs. Since we want
to count words, the words will be the keys and the value will be 1
indicating the presence of the word (to be summed from a list later to get
total count. """

def wc_mapper(document):
    """ generator that yields (word,1) for each word in document. 1
    indicates word presence """
    for word in tokenize(document):
        yield (word, 1)

""" 3. Write a reducer function that will produce outputs for each word. In
this case we will simply sum the word_presence_list. """

def wc_reducer(word, word_presence_list):
    """ sum up the word presences in word_presence_list """
    yield (word, sum(word_presence_list))

""" 2. Collect togehter all words that are identical """

def word_count(documents):
    """ Count the words in the input documents using MapReduce """
    
    # create dict container for holding word, presence_list pairs
    collector = defaultdict(list)
    
    # for each document call wc_mapper. As it generates (word,presence)
    # pairs add to the collector dict
    for document in documents:
        for word, presence in wc_mapper(document):
            collector[word].append(presence)

    # for each word, presence list pair in the collector call the reduce
    # yielding a tuple (word, sum(presence_list)) and return these tuples
    return [output for word, presence_list in collector.iteritems() 
            for output in wc_reducer(word, presence_list)]

# mini-test
print word_count(['data science', 'big data', 'science fun'])

""" So how would this work in practice? Imagine you had 100 machines. You
could do the following
1. Have each machine run wc_mapper on its set of documents producing (word,
   presence) pairs.
2. Give/Distribute those pairs to reducing machines making sure that pairs
   with the same word keys all end up at the same machine.
3. Have the reducing machines group the word, presence pairs into a dict
   collector keyed on the word with presence_list as values.
4. Run the reducer on each word, presence_list in the collector dict
5. Return each (key,counts) pair from all machines. """

# General MapReduce Approach #
##############################
# A more general framework of map reduce can be had by making minor
# alterations to the word counting code.

def map_reduce(inputs, mapper, reducer):
    """ runs MapReduce method on inputs using mapper and reducer
    functions"""
    collector = defaultdict([list])

    for input in inputs:
        for key, value in mapper(input):
            collector[key].append(value)

    return [output for key, value_list in collector.iteritems()
            for output in reducer(key, value_list)]


