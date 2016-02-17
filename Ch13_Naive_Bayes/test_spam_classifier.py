""" This module is for testing our Naive Bayes Spam Classifier. It opens a
set of emails from spamassasin.apache.org/publiccorpus and classifies the
subject lines as spam or not spam """

import glob
import re
import random

from collections import Counter
from DS_Scratch.Ch13_Naive_Bayes.spam_classifier import NaiveBayesClassifier
from DS_Scratch.Ch11_Machine_Learning import split_data, precision, recall


# path where spam/non-spam emails are stored
path = (r'/home/giladmeir/Python/Learn/Data_Science/'
        'DS_Scratch/Ch13_Naive_Bayes/spams/*/*')
data = []

# Get the (Subject, is_spam) pair #
###################################

# glob.glob gets the entire file path and name
for filename in glob.glob(path):
    # if ham is not anywhere in the file_path_name then is_spam = 1 else 0
    is_spam = "ham" not in filename

    with open(filename,'r') as file:
        for line in file:
            if line.startswith('Subject:'):
                # remove the leading 'Subject:' and keep left overs
                subject = re.sub(r'^Subject:','',line).strip()
                data.append((subject, is_spam)) 
                

# Split to train and test sets #
################################
random.seed(0)
train_data, test_data = split_data(data, 0.75)

# Call Classifier #
###################
classifier = NaiveBayesClassifier()
classifier.train(train_data)
# Remember this will assign word_probs = triplets of (word, P(x|S) and
# P(x|~S) to the classifier object needed for the classify method below

# return triplets of subject, is_spam binary and spam_probability for each
# email in test_data
classified = [(subject, is_spam, classifier.classify(subject)) for subject,
              is_spam in test_data]

# assume spam_probability > 0.5 corresponds to spam prediction and count
# number of actual is_spam to predicted is_spam in test data
counts = Counter((is_spam, spam_probability > 0.5) for _, is_spam,
                  spam_probability in classified)
print counts

# Calculate the precision and recall #
######################################
# Preecision measures the number of true positives to false positive (how
# precise are we in getting positive identification)
spam_precision = (counts[(True,True)]/
                       float((counts[(True,True)]+counts[(False,True)])))

print "Precision of spam classification is %.2f" %(spam_precision)

# recall measures how many we got right (true positives) to how many we
# missed (false negatives) 
spam_recall = (counts[(True,True)]/
                       float((counts[(True,True)]+counts[(True,False)])))
print " Recall of spam classification is %.2f" %(spam_recall)

# Examining Misclassifications #
################################
# Classified contains the subjects so we can examine what words are most
# likely classified as spam

# sort classified by spam_probability from smallest to largest
classified.sort(key=lambda row: row[2])
print "The spammiest subject lines:"
print classified[-5:]

# get the highest predicted spam probabilites from the non-spams
spammiest_non_spams = filter(lambda row: not row[1], classified)[-5:]
print "The spammiest non spams"
print spammiest_non_spams

# get the least likely spams
least_likely_spams = filter(lambda row: row[1], classified)[:5]
print "The least likely spams"
print least_likely_spams

# Get the spammiest words #
###########################
def p_spam_given_word(word_prob):
    """use Bayes Thm to calculate P(S| msg contains word)"""

    word, prob_if_spam, prob_if_not_spam = word_prob
    return prob_if_spam/(float(prob_if_spam)+float(prob_if_not_spam))

# recall word_probs is a list of triplets (x, P(x|s), P(x|~s)) stored as
# attribute of the classifier obj
words = sorted(classifier.word_probs, key=p_spam_given_word)

print "The spammiest words are"
print words[-5:]
