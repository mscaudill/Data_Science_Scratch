"""
In this module we will construct a Naive Bayes classifier for classifying
emails as spam or not spam based on specific words in the document. The
module will first start with a derivation of Bayes them and then build
assuptions to make a simple model for the classification. I will then
implement the algorithm.
"""

"""
Bayes Theorem

*************************
* A               #######*#############            
*                 #      *         B  # 
*                 # AinB *            # 
*                 #      *            # 
*                 #      *            #
*                 #      *            # 
*                 #######*#############             
************************

Consider the intersection of A and B (AinB). This area can be written as the
P(A|B)P(B) (the probability of getting A given that B has occurred). Now
this is equivalent to P(B|A)P(A). Setting these equal we can get 
P(B|A) = P(A|B)P(B)/P(A). This is Bayes Them. but we can go one step further
and rewrite P(A) = sum[P(A|B_i)*P(B_i)] if B_i's are disjoint.

In this module we will be looking at the probabiility an email is spam P(S)
given that it contains certain word(s). Lets say it contains the word viagra
then by Bayes Them
P(S|V) = P(V|S)*P(S)/[P(V|S)P(S)+P(V|~S)P(~S)]

ASSUMPTION 1: P(S) = P(~S) = 0.5: The probability is spam is 0.5
With this we can rewrite Bayes as:
P(S|V) = P(V|S)/[P(V|S)+P(V|~S)]

Now obviously we want to test not just for one word but for many words X_i.
This leads to our second assumption

ASSUMPTION 2: P(X_1, X_2, X_3,...|S) = P(X_1|S)*P(X_2|S)*... Independence
This is a major assumption because words indicating a spam document likely
occur together. This is why we call this Naive Bayes.
Now we can write Bayes them as a simple product of the probabilities

#################
product[P(S|X_i)] = product{P(X_i|S)/[P(X_i|S)+P(X_i|~S)]}
#################

Typically we do not mulitply probabilities directly because of underflow
with floating points so we will do the following with products of probs
P = P1*P2*P3*...
log(P) = log(P1)+log(P2)+...
P = exp(log(P)) = exp(log(P1)+log(P2)+....) That is we will sum logs of
probabilities.
Now all that is left is to estimate P(X_i|S) and P(X_i|~S). Example if the
word data did not appear in any spam messages then P('data'|S) = 0. This is
a problem becasue any msg contianing data would never be spam. So we bias
the probability with a pseudocount k
P(X_i|S) = (k + number of spams with word w_i)/(2k + number of spams). That
is we assume we saw k additional spams with the word and k additional spams
without the word. For example if data was in 0/98 documents and k=1 we
estimate P('data'|S) as (1+0)/(100) = 0.01 which allows the classifier to
assign a non-zero probability to documents containing 'data' word.
"""

# Implementation #
##################
from collections import Counter
from collections import defaultdict

import re
import math

# We first need to build a func to extract words from a message
def tokenize(message):
    # make all words lower case
    message = message.lower()
    # use findall to get only words
    all_words = re.findall("[a-z0-9']+", message)
    return set(all_words)

# We need to build a function that can count the words in a training set
# where the messages are known to be spam or not and then place the words as
# the keys and the values as [spam, not_spam] corresponding to how many
# times we saw that word in a spam and non-spam message.
def count_words(training_set):
    """ training set consist of pairs of (message,spam). Increments a 
        default dict with keys = words and values = [spam,not_spam] """
    counts = defaultdict(lambda: [0,0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts

# We now need to turn these word counts into probabilities P(X_i|S) and
# P(X_i|~S). We use the pseudocount smoothing estimate described above
def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    """ takes counts (a dictionary of words as keys and spam/non_spam as
        values) and converts this to a triplet w, P(w|spam), P(w|~spam) """
    return[(w, 
           (k + spams)/float(2*k+total_spams), 
           (k + non_spams)/float(2*k + total_non_spams)) 
           for w, (spams, non_spams) in counts.iteritems()]

# Last, we need to assign probabilities to the entire message
def spam_probability(word_probs, message):
    # get the words of the message
    message_words = tokenize(message)
    # initialize the log of the probability
    log_prob_if_spam = log_prob_if_not_spam = 0

    for word, prob_if_spam, prob_if_not_spam in word_probs:
        # if the *word* appears in the message add to the log prob of seeing
        # it in the message
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)

        # if the *word* does not appear in the message, we add to the log
        # probability of not seeing the word P(X_i~=x|S) = 1-P(X_i=x|S)
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)
    # return the Bayes them result P(S|X)
    return prob_if_spam/(prob_if_spam + prob_if_not_spam)

# Construct NaiveBayesClassifier class
class NaiveBayesClassifier:
    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, training_set):

        # count spam and non-spam messages
        num_spams = len([is_spam for message, is_spam in training_set
                        if is_spam])
        num_non_spams = len(training_set) - num_spams

        # run training data through our functions
        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(word_counts,num_spams,
                                             num_non_spams,self.k)

    def classify(self,message):
        return spam_probability(self.word_probs, message)
