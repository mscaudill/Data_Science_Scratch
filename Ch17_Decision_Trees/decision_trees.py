""" In this module 

"""

""" Given a set of data S with distict subsets C1...Cn the entropy is given
as H = -p1*log(p_1) - ....pn*log(p_n) where p_i is the probability that
subset c_i is occupied. If most p_i's = 0 then the entropy is low because
all the data is in one the classes."""

def entropy(class_probabilities):
    """ given a list of probabilities for different classes compute the
        entropy -- neglect classes with 0"""
    return sum(-p*math.log(p,2) for p in class_probabilities if p)

""" The data we have will consist of pairs (input, label) so we need to
compute the class probabilities."""

def class_probabilities(labels):
    """ returns list of probabilities one per class/subset of data"""
    total_count = len(labels)
    return [count / float(total_count) 
            for count in Counter(labels).values()]

def data_entropy(labeled_data):
    """ returns the entropy of the labeled_data """
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(class_probabilities)
