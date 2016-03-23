"""
Here we are going to build a statistical model of language from a given
document. The model works by using a given word in a document and choosing
n-pairs of words surrounding the given word and then forming sentences. We
will do this for the article "What is data science" on the Oreilly website
so just as in Ch9 we will use beautiful soup to parse the words and periods
"""

from bs4 import BeautifulSoup
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
