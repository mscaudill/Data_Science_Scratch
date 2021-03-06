""" In this module, we will explore a data set using a decision tree. Here
is the problem. You have a set of interviewees at a company and you have a
set of attributes for each interviewee. You also have a label of whether the
interview went well. This divides the data into two classes (True, False)
###########################
# c1=true  #  # c1=false  #
#       _  #_______       #
#       |  # Senior|      #
#       |T #   F  F|      #
###########################
The question is 'can we use the attributes to predict
whether a candidate will lie in the true of false class (i.e. have a good
interview?' The structure of the candidate data is 
[({'level': 'Senior', 'lang': 'Java', 'tweets': 'no','phd', 'no'}, False), 
(), ...] a list of tuples where the 0th el is an attributes dictionary and 
the 1st element is whether they interviewed well.

To address how the attributes contribute to interview success we will create
a decision tree. This works by iteritavely partitioning the data according 
to an attribute (we will call these partitions; See 'level' partition above)
and then measuring the entropy of the partitioned data set (Basically the 
log of the probability that the class is occuppied. Our goal is to partition
and repartition in such a way as to drive the entropy to 0. meaning we 
become certain as to whether the candidate gave falls into a particular 
class base on a set of partitioning attributes """

import math
from collections import Counter, defaultdict
from interviewee_data import data as candidates
from functools import partial

# Compute Unpartitioned Entropy #
#################################

# Here we will compute the entropy for the unpartitioned data. It simple
# gets the label for each candidate, computes the probabilities for being in
# the true and false class and then computes the entropy.

""" Given a set of data S with distict subsets C1...Cn the entropy is given
as H = -p1*log(p_1) - ....pn*log(p_n) where p_i is the probability that
subset c_i is occupied. If most p_i's = 0 then the entropy is low because
all the data is in one the classes. In this example, we have two classes
true and false """
def entropy(class_probabilities):
    """ given a list of probabilities for different classes compute the
        entropy -- neglect classes with 0"""
    return sum(-p*math.log(p,2) for p in class_probabilities if p)

""" The candidate data tells us how many candidates fall into each class. We
can use this info along with the total number of candidates to compute the
probability of falling into either the true or false class """
def class_probabilities(labels):
    """ returns list of probabilities one per class/subset of data"""
    total_count = len(labels)
    return [count / float(total_count) 
            for count in Counter(labels).values()]

""" Lastly given the probabilities of each candidate being in either the
true or false class we can compute the entropy of all the candidates """
def data_entropy(candidate_data):
    """ returns entropy of the candidate_data according to the classes """
    # get label for each candidate in candidate data
    labels = [label for _, label in candidate_data]
    #print labels
    # compute the probability of being in true or false classes
    probabilities = class_probabilities(labels)
    # return the entropy of the candidate data
    return entropy(probabilities)

# Compute Partitioned Entropy #
###############################
# So far we have simply computed the total entropy of the candidate data.
# Now we will partition the data by a specific attribute and recompute the 
# entropy.

def partition_by(inputs, attribute):
    """ each candidate_data(input) is a tuple (attribute_dict, label). 
    Returns a list of dictionaries keyed on attribute. ex if level chosen 
    then three dicts returned [Senior, Mid, Junior] """
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]
        groups[key].append(input)
    return groups

def partition_entropy(subsets):
    """ Computes the entropy following a specific partition of labeled data.
    note it is simply a weighted sum of the entropies of each partition
    where the scale factor is the size of the partition. In this example it
    if the attribute was the level then returns a weighted sum of senior,
    mid and junior partitions"""
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / float(total_count)
               for subset in subsets)

def partition_entropy_by(inputs, attribute):
    """ computes entropy for a given partition defined by the attribute """
    # partition the data by the attribute
    partitions = partition_by(inputs, attribute)
    # compute the entropy of the partitioned data
    return partition_entropy(partitions.values())

# GENERALIZED DECISION TREE #
#############################
# Here we build a generalized algorithm for a decision tree. The tree will
# be madeup of the following:
# True - a leaf node that returns True for any input
# False - a leaf node that returns False for any input
# a tuple (attribute, subtree_dict) which is a decision node in our tree.

# In our example this means our tree would be represented as
'''tree = ('level',{'Junior': ('phd', {'no' : True, 'yes' : False}), 
             'Mid' : 'True',
             'Senior' : ('tweets' : {'yes' : True, 'no' : False})})'''

# Given such a representation we can classify an input candidate. This
# candidate is represent by a dict like this one 
""" input = { 'level': 'Junior', 'lang' : 'Java', 'tweets' : 'yes', 
      'phd' : 'no'} """

# Classify Input #
##################
# Given a tree like the example one above and a candidate we want to
# classify the candidate as true or false for a good or bad interview.
def classify(tree, input):
    """ classify an input using the given decision tree """
    
    # if this is a leaf node i.e. True or False return its value
    if tree in [True, False]:
        return tree

    # otherwise this is a decision node (like Junior above). We assign this
    # to a subtree tuple '( 'phd', {'no' : True, 'yes' : False} )
    attribute, subtree_dict = tree

    # Now we get the key of the subtree (it is just the 
    # input.get('phd') == 'no'. Will return None if attribute not in input
    subtree_key = input.get(attribute)

    if subtree_key not in subtree_dict:
        subtree_key = None
    
    # The subtree is then subtree_dict value for the subtree key
    # (subtree['no'] = False
    subtree = subtree_dict[subtree_key] 
    # classify the inpput using this subtree until we hit a leaf. In our
    # example we have hit a false leaf so the classification yields false on
    # the next pass of this function
    return classify(subtree, input)

# Build tree representation #
#############################
# Now that we can classify an input we need to build a tree like the
# example one above. Again the structure of the candidate data is initially
# [({'level': 'Senior', 'lang': 'Java', 'tweets': 'no','phd' : 'no'}, 
#   False), (), ...] a list of tuples where the 0th el is an attributes 
# dictionary and the 1st element is whether they interviewed well.
def build_tree_id3(inputs, split_candidates=None):
    # if this is our first pass through then all attributes are a splitting
    # candidate
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()
    
    # count the trues and falses in inputs
    num_inputs = len(inputs)
    num_trues = len([label for _, label in inputs if label])
    num_falses = num_inputs - num_trues

    # if there are no true values we return a false leaf
    if num_trues == 0:
        return False
    # conversely if only trues return a true leaf
    if num_falses == 0:
        return True

    # If we have run out of splitting candidates, return the majority leaf
    if not split_candidates:
        return num_trues >= num_falses

    # otherwise split on the best attribute
    best_attribute = min(split_candidates, 
                         key=partial(partition_entropy_by, inputs)) 

    # partition the data by the best attribute and get the new candidates
    partitions = partition_by(inputs, best_attribute)
    new_candidates = [attribute for attribute in split_candidates
                      if attribute != best_attribute]

    # recursively build the subtrees
    subtrees = {attribute_value : build_tree_id3(subset, new_candidates)
                for attribute_value, subset in partitions.iteritems()}

    subtrees[None] = num_trues >= num_falses

    return (best_attribute, subtrees)

# Random Forest #
#################
""" Decision trees have a tendency to overfit training data and so one
technique to deal with this is to create a random forest in which we build
multiple decision trees and let them vote on how to classify inputs """
def forest_classify(trees, input):
    votes = [classify(tree, input) for tree in trees]
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]

""" The trees we have made so far have been deterministic because they are
based on choosing the next attribute with the lowest entropy. How can we
build random trees? Below are two methods """
# method one is to bootstrap the candidate data. Since each sample will be
# different, we will get different trees. This method is call bootstrap
# aggregating or bagging.

# method two is split on the best attribute from a random selection of
# all the attributes. This method is called ensemble learning.

if __name__ == '__main__':
    
    # Iniitially we do not know which attribute to partition the data on, so
    # we will go through all the attributes and locate the attribute that
    # yields the lowest entropy in the first step. This is called a greedy
    # algorithm because it chooses the immediate best option. But note this
    # may not yield the optimal tree, a bad first step may ultimately lead
    # to a lower overall entropy.
    for key in ['level', 'lang', 'tweets', 'phd']:
        print key, partition_entropy_by(candidates, key)
        # Results
        #level 0.693536138896
        #lang 0.860131712855
        #tweets 0.788450457308
        #phd 0.892158928262


    # The lowest entropy results from partitioning by the level so we need
    # to make a subtree for senior, mid and junior levels and recompute
    # entropy
    senior_inputs = [(input, label) for input, label in candidates 
                     if input['level'] == 'Senior']

    print 'Senior Subtree Entropies ------------------------'
    for key in ['lang', 'tweets', 'phd']:
        print key, partition_entropy_by(senior_inputs, key)

    junior_inputs = [(input, label) for input, label in candidates 
                     if input['level'] == 'Junior']

    print 'Junior Subtree Entropies ------------------------'
    for key in ['lang', 'tweets', 'phd']:
        print key, partition_entropy_by(junior_inputs, key)
    
    mid_inputs = [(input, label) for input, label in candidates 
                     if input['level'] == 'Mid']
    
    print 'Mid-level Subtree Entropies ---------------------'
    for key in ['lang', 'tweets', 'phd']:
        print key, partition_entropy_by(mid_inputs, key)

    print 'Generalized Tree and Classification--------------'    

    # Generalized Tree Build and Classification #
    #############################################
    # Build our tree from the candidate data
    tree = build_tree_id3(candidates)
    print 'Decision Tree'
    print tree
    # classify a candidate
    print 'Decision Tree Classification'
    print classify(tree, {'level' : 'Junior', 'lang' : 'Java',
                          'tweets' : 'yes', 'phd' : 'no'})

