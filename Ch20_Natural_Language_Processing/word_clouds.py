"""
In this small module we will show how to make a word cloud from a data set
of tuples with three els. el 1 is the word, el2 is the popularity on job
postings and el3 is the popularity on resumes. """

from collections import defaultdict
from matplotlib import pyplot as plt

# Fake data from the book
data = [ ("big data", 100, 15), ("Hadoop", 95, 25), 
         ("Python", 75, 50), ("R", 50, 40), 
         ("machine learning", 80, 20), ("statistics", 20, 60),
         ("data science", 60, 70), ("analytics", 90, 3),
         ("team player", 85, 85), ("dynamic", 2, 90),
         ("synergies", 70, 0), ("actionable insights", 40, 30), 
         ("think out of the box", 45, 10), ("self-starter", 30, 50),
         ("customer focus", 65, 15), ("thought leadership", 35, 35)]

counts= defaultdict(int)

def text_size(word):
    """ alters the size of the text in the word cloud based on count. It
    maps the total word counts to a number bewteen 8 and 28. """
    
    for word, job_popularity, resume_popularity in data:
        counts[word] = job_popularity + resume_popularity
    
    min_count = min(counts.values())
    max_count = max(counts.values())

    return min(8, min_count) + counts[word]/float(max_count) * 28

for word, job_popularity, resume_popularity in data:
    plt.text(job_popularity, resume_popularity, word, ha='center',
             va='center', size = text_size(word))

plt.xlabel('Popularity on Job Postings')
plt.ylabel('Popularity on Resumes')
plt.axis([0, 100, 0, 100])
plt.xticks([])
plt.yticks([])
plt.show()
