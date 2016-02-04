"""
In this module we will look at a sample data set generated from a binomial
distribution and look at some statistics that describe the data
"""
from numpy import random
from matplotlib import pyplot as plt
from collections import Counter
from Ch4_Linear_Algebra import sum_of_squares

import math

############################################################################
# Create binomially distributed data
############################################################################
"""
The binomial distribution follows P_x(k) = (n choose K)*p^k*(1-p)^(n-k). The
thought exp is flipping a coin n times and asking what is the probability of
k successes where each flip has probability p for heads
"""
n = 1000 # 1000 bernouilli trials
p = 0.25 # probability of heads for a given trial
size = n # draw out the same number of samples as their are trials
# call binomial to make the distribution
successes = random.binomial(n,p,size)

############################################################################
# Plot the distribution of heads for our 1000 trials
############################################################################
# Use Counter to count the number of heads
num_heads = Counter(successes)
# The x-axis of the number of k heads possible (0,1000)
xs = range(0,n)
# the ys is the number of times k heads occurs
ys = [num_heads[x] for x in xs]
# set up a new figure
plt.figure(1)
# make a bar plot
plt.bar(xs,ys)

############################################################################
# Mean
############################################################################
"""
The first statistic is the mean. For a binomial the mean can be calculated
because the binomial is a sum of indpt bernoulli trials and by linearity of
Expectation E[ x1+x2+x3+...] = E[x1]+E[x2]+.... = np since each has prob=p
"""
def mean(x):
    """ returns the sample mean of x """
    return sum(x)/float(len(x))

# print the sample mean of the binomial distribution above
print "The mean number of heads is %f" %(mean(successes))

############################################################################
# Median
############################################################################
def median(x):
    """ returns the middle-most sorted value of x """
    # get length of x and sort from low to high and midpt
    n = len(x)
    sorted_x = sorted(x)
    midpt = n // 2 # force int result (rounds down 9//2 = 4)
    
    # test for even or odd number of elements in x
    if n % 2 == 1:
        # if odd return the middle value
        return sorted_x(midpt)
    if n % 2 == 0:
        # if even return the mean of the middle two values
        lo = midpt-1
        hi = midpt
        return (sorted_x[hi] + sorted_x[lo])/float(2)

print "The median value is %f" %(median(successes))

############################################################################
# Quantile
############################################################################
"""
The quantile is the data point below which a certain percentage of the data
lies. Ex. the median is the data pt where 50% of the data lies below
"""

def quantile(x):
    """ returns the pth-percentile value in x """
    p_index = int(p*len(x))
    # ex if len(x) = 100 the 10% point in x is P_index=10
    return sorted(x)[p_index]

############################################################################
# Dispersion Metrics
############################################################################

############################################################################
# Data Range
############################################################################
def data_range(x):
    return max(x)-min(x)

############################################################################
# Variance
############################################################################
"""
Variance is a measure of how spread out the data is relative to the mean
E[(X-E[X])^2]. We approx this by 1/n*sum_k((x_k-xbar)^2). But this is biased
so we change 1/n to 1/n-1 to give an unbiased estimator of variance
"""
# de-mean
def demean(x):
    """ translates x by subtracting its mean """
    return [x_i - mean(x) for x_i in x]

def variance(x):
    n = float(len(x))
    return sum_of_squares(demean(x))/(n-1)

print "The variance in the number of heads is %f" %(variance(successes))

def standard_deviation(x):
    """ Computes the standard deviation of x """
    std = math.sqrt(variance(x))
    return std
print "The standard deviation in the number of heads is %f" 
        %(standard_deviation(successes))

############################################################################# Covariance and Correlation
############################################################################
""" Covariance measures how two random variables X & Y vary togehter. 
It is defined as E[(x-E[x])(y-E[y]). As usual we approximate this 
sum((x_k-x_bar)*(y_k-y_bar))/(n-1)
"""

def covariance(x,y):
    n = len(x)
    return dot_product(demean(x),demean(y))/(n-1)

""" correlation is the covariance standardized (x-x_bar)/std(x) """
def correlation(x,y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x and stdev_y > 0:
        return covariance(x,y)/(stdev_x*stdev_y)
    else:
        return 0

plt.show()
