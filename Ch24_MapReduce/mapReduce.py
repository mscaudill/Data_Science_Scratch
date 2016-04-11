"""
MapReduce is a programming model for performing parallel processing on large
data sets. There are essentially three steps to mapReduce.
1. use a mapper func to turn each item into (key, value) pairs.
2. collect together all the pairs with identical keys into groups (dict obj)
3. use a reducer function to produce outputs for each key
"""
from DS_Scratch.Ch13_Naive_Bayes.spam_classifier import tokenize
from collections import defaultdict
from datetime import date

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

# with this framework we can abstract the types of mapper and reducer
# functions. Previously the reducer simply summed the values list but we can
# more generally write this using an aggregate function

def reduce_values_with(aggregation_fn, key, value_list):
    """ applies aggregation function to values yielding a (key,output)
    pairs """
    yield (key, aggregation_fn(value_list))

def values_reducer(aggregation_fn):
    """ turns an aggregation func that maps value_list -> output into a 
    reducer mapping (key, value_list) -> (key, output) tuple """
    return partial(reduce_values_with(aggregation_fn, key, value_list))

# now for example we can write 
sum_reducer = values_reducer(sum)
max_reducer = values_reducer(max)
min_reducer = values_reducer(min)
count_distinct_reducer = values_reducer(lambda values_list:
                                        len(set(values_list)))
# and so on

# Example: Analyzing Status Updates #
#####################################
""" 
Given a status updates, we can use map_reduce to address questions like.
What days of the week are people most likely to be talking about data
science? Here is an implementation...
"""
 status_updates = {'id':1, 'username':'mscaudill', 'text':'read any good data
 science books lately?','created at': datetime.datetime(2013, 12,
 21,11,47,0, 'liked_by' : ['joelgrus','data_dude','data_gal']}

# Given a status update like the one above can we figure out what day of the
# week people talk most about data science? We can use the day of the week
# as a key to a mapper and then the value as one for each update that falls
# on that day. Then simply sum them up.

def data_science_day_mapper(status_update):
    """ yields a (day_of_week,1) tuple if a tweet appears discussing data
    science on that day """
    if 'data science' in status_update['text'].lower():
        day_of_week = status_update.weekday()
        yield (day_of_week, 1)

# now call our general implementation of map_reduce using the sum_reducer
data_science_days = map_reduce(status_updates, data_science_day_mapper,
                               sum_reducer)

""" Now consider a slightly more complicated example. For a given user what
is the most common word in their status updates? To implement this we will
key on the username and the values will be the word and counts for that
word"""


