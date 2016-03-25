"""
Here we are going to build a statistical model of language from a given
document. The model works by using a given word in a document and choosing
n-pairs of words surrounding the given word and then forming sentences. We
will do this for the article "What is data science" on the Oreilly website
so just as in Ch9 we will use beautiful soup to parse the words and periods.
We will then explore Grammars which are rules for generating acceptable
sentences. Finally, we will look at Topic modeling using the Latent
Dirichlet Allocation. This is a very interesting topic that needs further
exploration than the text provides. Wikipedia seems like a good starting
place.
"""

from bs4 import BeautifulSoup
from collections import defaultdict, Counter
import random
import requests
import re
from tabulate import tabulate

# Website of the document
url = "http://radar.oreilly.com/2010/06/what-is-data-science.html"
# use request to get the html of the url
html = requests.get(url).text
# call beautiful soup so we can look up the "parts" of the html
soup = BeautifulSoup(html, 'html5lib')
# look for the div containing the article body
content = soup.find("div", "article-body")
# Use regexp to get words and periods
regex = r"[\w']+|[\.]"

document = []

def fix_unicode(text):
    return text.replace(u"\u2019", "'")

for paragraph in content("p"):
    words = re.findall(regex, fix_unicode(paragraph.text))
    document.extend(words)

# Bigrams #
###########
# Bigrams are n-grams where we use the first word following a starting word
# to pick a starting word we just choose a random word following a period.

bigrams = zip(document, document[1:])
# we will create a dict that for each word will hold as values all the
# words that follow this word in the document
transitions = defaultdict(list)
for prev, current in bigrams:
    transitions[prev].append(current)

# now we can generate sentences
def bigram_sentence_generator():
    current = "."
    result = []
    while True:
        # look up in transitions dict the next word
        next_word_candidates = transitions[current]
        # make the new current word a random choice from the candidates
        current = random.choice(next_word_candidates)
        # append to results to build a sentence
        result.append(current)
        if current == ".": 
            bigram_sentence = " ".join(result)
            return bigram_sentence

# Trigrams #
############
# The bigram sentence is gibberish, we can make it less so by considering
# three consecutive words called a trigram
trigrams = zip(document, document[1:], document[2:])
trigram_transitions = defaultdict(list)
starts = []

for prev, current, next in trigrams:
    # here we need to keep track of starts because we need to get the next
    # two words so we can't just make a random choice from the next word
    # candidates as we did before
    if prev == '.':
        starts.append(current)

    trigram_transitions[(prev,current)].append(next)

# now we can generate trigram sentences
def trigram_sentence_generator():
    current = random.choice(starts)
    prev = '.'
    result = [current]

    while True:
        next_word_candidates = trigram_transitions[(prev, current)]
        next_word = random.choice(next_word_candidates)

        prev, current = current, next_word
        result.append(current)

        if current == '.':
            return " ".join(result)

# Grammars #
############
# Here we will define a set of grammar rules. Underscores are rules that
# need further expanding and other names are terminals that do not need
# further expanding. We generate sentences by repeatedly replacing each rule
# with its child values ex. _S -> _NP _VP a noun phrase and verb phrase ->
# _N _V which goes to a noun and verb
grammar = {
        "_S"  : ["_NP _VP"],
        "_NP" : ["_N",
                 "_A _NP _P _A _N"],
        "_VP" : ["_V",
                 "_V _NP"],
        "_N"  : ["data science", "Python", "regression"],
        "_A"  : ["big"", linear", "logistic"],
        "_P"  : ["about", "near"],
        "_V"  : ["learns", "trains", "tests", "is"]}


def is_terminal(token):
    return token[0] != '_'

def expand(grammar, tokens):
    for i, token in enumerate(tokens):
        # skip over terminals
        if is_terminal(token):
            # continue in next for iteration
            continue

        # if not a terminal we need to choose a replacement
        replacement = random.choice(grammar[token])

        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            # splice the non-terminal token list  into tokens
            tokens = tokens[:i] + replacement.split() + tokens[(i+1):]
        
        # now call expand on this new list of tokens 
        return expand(grammar, tokens)

    # if we get here they are all terminals and we return tokens
    return tokens

# Topic Modeling #
##################
# We are interested in identifying common topics across a set of documents.
# One method for doing this is called the Latent Dirichlet Analysis (LDA).
# It shares some similarities with Naive Bayes Classification. Here are the
# assumptions of the model:
# 1. There are k fixed topics
# 2. There is a random variable that assigns each topic an associated prob.
# distribution over words. This dist is prob of seeing word w given topic k
# 3. There is another rv that assigns each document a probability
# distribution over topics. This distribution is the mixture of topics k in
# document d
# Each word in document d was generated by randomly picking a topic from the
# documents distribution of topics and then randomly picking a word from the
# topic's distribution of words
# the 4th documents 5th word is documents[3][4] and the topic of this word
# is document_topics[3][4]
# We will estimate the probability that a topic produces a certain word by
# looking at how many times that topic produces any word. We will use Gibbs
# sampling to generate document topics.
# Algorithm
# 1. assign to every word in d a random topic.
# 2. go through every word in each document. Construct weights for each
# topic tha depend on the current distribution of topics for the document
# and the current distribution of words for that topic.
# 3. Use the weights to sample a new topic for that word
# 4. By iterating the above we end up with a joint sample from th e
# topic-word distribution and document-topic distribution


documents = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels","pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data",
    "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce","Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence","probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vectormachines"]
    ]

# try to locate K=4 topics
K = 4

def sample_from(weights):
    """ returns an index i, with probability weights[i]/sum(weights) """
    total = sum(weights)
    rnd = total * random.random()
    for i, w in enumerate(weights):
        # return the smallest i such that weights[0] + ...weights[i] >= rnd
        rnd -= w
        if rnd <= 0: 
            return i

## Construct Containers for word counts, topic counts ##

# Count how many times each topic is assigned to a document
document_topic_counts = [Counter() for _ in documents]

# Count how many times a word is assigned to each topic for K topics
topic_word_counts = [Counter() for _ in range(K)]

# Initialize a list of how many total words are assigned to each topic
topic_counts = [0 for _ in range(K)]

# get the total number of words in each document
document_lengths = map(len,documents)

# get the number of distinct words
distinct_words = set(word for document in documents for word in document)
W = len(distinct_words)

# get the number of documents
D = len(documents)

## Define Conditional Probability Functions ##
def p_topic_given_document(topic, d, alpha=0.1):
    """ fraction of words in document d assigned to topic plus smoothing """
    # smoothing ensures every word has a non-zero chance of being chosen for
    # any topic
    return ((document_topic_counts[d][topic] + alpha)/
            (document_lengths[d] + K * alpha))

def p_word_given_topic(word, topic, beta=0.1):
    """ fraction of words assigned to a topic plus some smoothing """
    return ((topic_word_counts[topic][word] + beta)/
            (topic_counts[topic] + W * beta))

# create the weights for updating the topics 
def topic_weight(d, word, k):
    """ given document and word return weight for kth topic """
    return p_word_given_topic(word, k) * p_topic_given_document(k, d)

# perform weighted sample from the topics
def choose_new_topic(d, word):
    return sample_from([topic_weight(d, word, k) for k in range(K)])

# Implementation #
##################
# Initialiaze a random topic for each word in each document
random.seed(0)
document_topics = [[random.randrange(K) for word in document] for
                    document in documents]

for d in range(D):
    for word, topic in zip(documents[d], document_topics[d]):
        document_topic_counts[d][topic] += 1
        topic_word_counts[topic][word] += 1
        topic_counts[topic] += 1

# our goal is to get a joint sample from the conditional probabilities
# defined earlier we do this using Gibbs sampling.
for iter in range(1000):
    for d in range(D):
        for i, (word,topic) in enumerate(zip(documents[d],
                                             document_topics[d])):

            # remove this word/topic so it doesn't influence weights
            document_topic_counts[d][topic] -= 1
            topic_word_counts[topic][word] -= 1
            topic_counts[topic] -= 1
            document_lengths[d] -= 1

            # choos a new topic based on the weights
            new_topic = choose_new_topic(d, word)
            document_topics[d][i] = new_topic

            # now add it back into the counts
            document_topic_counts[d][new_topic] += 1
            topic_word_counts[new_topic][word] += 1
            topic_counts[new_topic] += 1
            document_lengths[d] += 1




if __name__ == '__main__':
    
    print '\n' 
    print "BIGRAM SENTENCE --------------"
    print bigram_sentence_generator()
    print '\n'

    print "TRIGRAM SENTENCE -------------"
    print trigram_sentence_generator()
    print '\n'

    print "GRAMMARS SENTENCE ------------"
    print expand(grammar, ["_S"])
    print '\n'

    print "TOPIC MODELING ---------------"

    # print out the topics
    table = []
    for k, word_counts in enumerate(topic_word_counts):
        for word, count in word_counts.most_common():
            if count > 0:
                table.append([k, word, count])
    print tabulate(table, headers=['Topic', 'Word', 'Count'])

