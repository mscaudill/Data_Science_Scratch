"""
In this module we explore conditional probabiility, various distributions,
Bayes them and the centeral limit them
"""
import numpy as np
from matplotlib import pyplot as plt
import math
from collections import Counter

###########################################################################
# Conditional Probability
############################################################################
"""We start by thinking of the common example of the probability of getting 
girl within a 2 child set knowing that at least one of them is a girl. The 
answer we have worked out in Probability stattistics and random processes 
 book long ago is 1/3. Lets show this computationally"""

def random_kid():
    """ a function to randomly choose boy or girl for many families"""
    return np.random.choice(["boy","girl"])

# initialize the number of girls
both_girls = 0
older_girl = 0
either_girl = 0

np.random.seed(0)
for _ in range(10000):
    # call rand_kid to select the sex of older and younger
    younger = random_kid()
    older = random_kid()
    # increment the appropriate girl case
    if older == "girl":
        older_girl += 1
    if older == "girl" and younger == "girl":
        both_girls += 1
    if older == 'girl' or younger == 'girl':
        either_girl += 1

# print out the probabilities
print "P(both|older): ", both_girls / float(older_girl)
print "P(both|either): ", both_girls / float(either_girl)

############################################################################
# Uniform distribution
############################################################################
def uniform_pdf(x):
    """ The probability density function of the uniform dist """
    if x > 0 and x < 1:
        return 1
    else: return 0

def uniform_cdf(x):
    """ The cumulative distribution func of the Uniform dist """
    if x > 0 and x < 1:
        return x
    elif x > 1:
        return 1
    else: return 0

############################################################################
# Normal Distribution
############################################################################
"""The normal distribution is the 'king' distribution. The reason is the 
CLT. The CLT says that given a set of i.i.d. random vars. *from any 
distribution* the standard value Z = (X_bar-mu)/(sigma/sqrt(n)) converges 
in distribution to the standard normal distribution for large n"""

def normal_pdf(x,mu=0,sigma=1):
    """ The pdf of the normal distribution with mean mu and std sigma"""
    root_two_pi = math.sqrt(2*math.pi)
    return 1/(root_two_pi*sigma)*math.exp(-(x-mu)**2/(2*sigma**2))

# Lets make a few plots
xs = [x/10.0 for x in range(-50,50)]
plt.figure(1)
plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0, sigma=1')
plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0, sigma=2')
plt.plot(xs,[normal_pdf(x,sigma=.5) for x in xs],':',label='mu=0,sigma=.5')
plt.plot(xs,[normal_pdf(x,mu=-1) for x in xs],'-.',label='mu=-1, sigma=1')
plt.legend()
plt.title('Various Normal PDFs')

# The CDF of the standard normal distribution is important because we can use it to calculate thresholds (p-values), confidence intervals etc to understand how well we may be estimating model parameters such as mu or sigma etc...
def normal_cdf(x, mu=0, sigma=1):
    """ Constructs the nomal cumulative distribution function """
    return(1+math.erf((x-mu)/(math.sqrt(2)*sigma)))/2
xs = [x/10.0 for x in range(-50,50)]
plt.figure(2)
plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0, sigma=1')
plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0, sigma=2')
plt.plot(xs,[normal_cdf(x,sigma=.5) for x in xs],':',label='mu=0,sigma=.5')
plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1, sigma=1')
plt.legend()
plt.title("Various Normal CDFs")

############################################################################
# Binary search algorithm for inverse cdf
############################################################################
"""Recall that the normal CDF gives the probability that an observation Z o
a random variable is < z. What we want to do is to approximate the inverse 
of this function. This will be helpful for determing quantities like
significane. Most of the time you can use builtin functions like ppf 
(perrcent point function). Here we write our own routine for practice."""

def inverse_normal_cdf(p ,mu=0 ,sigma=1, tolerance =0.0001):
    """ Finds the inverse of the normal cdf given a probability p, mean, 
        stand tolerance using a binary search algorithm """
    # if not standard, compute standard variables and rescale
    if mu != 0 or sigma != 1:
        return  mu + sigma*inverse_normal_cdf(p, tolerance=tolerance)

    # set the starting values of z's and probabilities p a cdf( -10)~0
    low_z, low_p = -10, 0 
    # do the same for the high values where cdf(10)~1
    high_z, high_p = 10, 1
    # Begin binary algorithm
    while high_z - low_z > tolerance:
        mid_z = (high_z + low_z)/float(2)
        # for this new z value calculate the CDF
        mid_p = normal_cdf(mid_z)
        # if the new mid_p value is below the supplied p then increase 
        #low_z and low_p to mid_z and mid_p
        if mid_p < p:
            low_z, low_p = mid_z, mid_p
        # else if the new mid_p is larger than supplied p reduce hi_z and
        # hi_p
        elif mid_p > p:
            high_z, high_p = mid_z, mid_p
        else:
            break
    return mid_z

print "The z value for probability 0.75 is %f" %(inverse_normal_cdf(0.95))

############################################################################
# Central limit Thm
############################################################################
"""As we have learned from probability theory if we have a large collection 
of random variables X_i then the standard variable 
Z = (X_bar-mu)/(sigma/sqrt(n)) will converge in distribution to a standard 
normal distribution. For example lets say we have many bernouilli trials 
with E[X] = p and var(X)= p(1-p). The sum of the iid variables is a 
binomial Y(n,p) and Z = (Y(n,p)-np)/sqrt(np(1-p)) is approx normal. So the 
distribution converges to a normal distribution with mean np and 
var = np(1-p)"""

def bernoulli_trial(p):
    """ Performs a single bernoulli trial with success p """
    return 1 if np.random.random() < p else 0

def binomial(n,p):
    """ returns the sum of n bernoulli trials """
    return sum(bernoulli_trial(p) for _ in range(n))

# now the central limit thm says that as n gets large binomial(n,p) 
# approaches a normal distribution with mean np and std = sqrt(np(1-p))
# lets plot to see that this is the case
def make_hist(p, n, num_points):
    """ makes a histogram of binomial data and normal data to show the truth    of the central limit theorem"""
    data = [binomial(n,p) for _ in range(num_points)]
    # use a bar chart to show the distribution of bin(n,p)
    histogram = Counter(data)
    plt.figure(3)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v/float(num_points) for v in histogram.values()],
            0.8, color = '0.75')
    
    # now make the normal distribution
    mu = n*p
    sigma = math.sqrt(n*p*(1-p))

    # make a line chart to overlay onto the binomial distribution
    xs = range(min(data), max(data)+1)
    ys = [normal_pdf(x,mu,sigma) for x in xs]
    plt.plot(xs,ys)
    plt.title("Binomial Distribution vs. Normal Approximation")

make_hist(0.75,100,1000)
# Only show these plots if being called as main
if __name__ == "__main__":
    plt.show()
