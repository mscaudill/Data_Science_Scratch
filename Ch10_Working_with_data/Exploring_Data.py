"""
Code sample describing how to work with data from Ch 10 of Data Science from
Scratch by J. Grus 2014
"""

import math
import random

from collections import Counter
from matplotlib import pyplot as plt
from Ch4_Linear_Algebra import shape, get_col
from Ch6_Probability import inverse_normal_cdf

# Bucketed Histograms #
#######################
def bucketize(point, bucket_size):
    """floor the point to the next lower multiple of bucket size"""
    return bucket_size * math.floor(point/bucket_size)

def make_histogram(points, bucket_size):
    """ buckets the points and counts how many in each bucket"""
    return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points, bucket_size, title=''):
    """ makes a histogram of bucketed points"""
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(),histogram.values(), width = bucket_size)
    plt.title(title)
    plt.show()

# for example consider the uniform distribution
random.seed(0)

uniform = [200*random.random()-100 for _ in range(10000)] 
#plot_histogram(uniform, 10, title = "Uniform Histogram")

# Two-dim Data #
################

# consider two-dim data
def random_normal():
    """ returns a random draw from normally distributed data"""
    return inverse_normal_cdf(random.random())

xs = [random_normal() for x in range(1000)]
ys1 = [x + random_normal() / 2.0 for x in xs]
ys2 = [-x + random_normal() / 2.0 for x in xs]

plt.scatter(xs,ys1, marker='.', color = 'black', label = 'ys1')
plt.scatter(xs,ys2, marker='.', color = 'gray', label = 'ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title("Very different joint distributions")
plt.show()

# Multiple Dims #
#################

# A first pass approach to seeing how variables relate is to compute the
# correlation matrix

def correlation_matrix(data):
    """ returns a num_cols x num_cols matrix where the (i,j)th entry is the
    correlation between columns i and columns j of data"""
    
    _, num_columns = shape(dta)
    def matrix_entry(i,j):
        return correlation(get_col(data,i),get_col(data,j))

    correlation_matrix =  make_matrix(num_cols,num_cols,matrix_entry)
    return correlation_matrix

# a more visual approach is to look at a the scatter plot matrices for i,j
# see text for the details of this plotting. Its super easy.

