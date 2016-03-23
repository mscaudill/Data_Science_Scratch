"""
Here we are going to build a statistical model of language from a given
document. The model works by using a given word in a document and choosing
n-pairs of words surrounding the given word and then forming sentences. We
will do this for the article "What is data science" on the Oreilly website
so just as in Ch9 we will use beautiful soup to parse the words and periods.
We will then explore Grammars which are rules for generating acceptable
sentences.
"""

from bs4 import BeautifulSoup
from collections import defaultdict
import random
import requests
import re

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

print "BIGRAM SENTENCE ------------"
print bigram_sentence_generator()
print '\n'

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

print "TRIGRAM SENTENCE --------------"
print trigram_sentence_generator()


