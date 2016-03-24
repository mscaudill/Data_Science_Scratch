"""
In this brief module I will look at Gibbs sampling. This is a deep topic
that deserves greater attention. In short, Gibbs Sampling is a Markov Chain
Monte Carlo algorithm method for obtaining samples from a mulivariate
distribution when direct sampling may be very difficult using conditional 
distributions.
"""
from collections import defaultdict
import random

# Example of Dice Rolls #
#########################
# Imagine that we are rolling two dice. Let x be the value of the first and
# y be the sum of the two dice and imaging you want to generate lots of
# (x,y) pairs.

def roll_a_die():
    return random.choice([1,2,3,4,5,6])

def direct_sample():
    d1 = roll_a_die()
    d2 = roll_a_die()
    return d1, d1 + d2

# now imagine you only knew the conditional distributions. P(Y|X) is easy
# because the values of y are equally likely to be x+1, x+2, ... x+6
def random_y_given_x(x):
    """ equally likely to be x+1, x+2, ....x+6 """
    return x + roll_a_die()

# The other conditional probability P(X|Y) is harder. For example if Y is 3,
# x could only be a 1 or a 2
def random_x_given_y(y):
    if y <= 7:
        # if the sum is less than 7, the first die is equally likely to be
        # 1, 2,.. y-1
        return random.randrange(1, y)

    else:
        # if the sum is more than 7, then the first die is equally likely to
        # be (total-6), (total-5)....,6
        return random.randrange(y-6, 7)

# Gibbs sampling Algorithm #
############################
# 1. Start with a valid value of (x,y)
# 2. Replace x with a value drawn from P(X|Y)
# 3. Replace y with a value drawn from P(Y|X)
# 4. Repeat steps 2 and 3
# The resulting (x,y) pair will represent a sample from the unconditional
# joint distribution P(x,y)

def gibbs_sample(num_iters=100):
    """ generates x,y pair from a joint distribution where only the
    conditional probabilities are known. """
    # provide a valid starting pair
    x, y = 1, 2 
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x,y

# Compare Direct Samples to Gibbs Sample #
##########################################

def compare_distributions(num_samples = 1000):
    counts = defaultdict(lambda: [0,0])
    for _ in range(num_samples):
        # assign gibbs sample and increment the number of times it occurs
        counts[gibbs_sample()][0] += 1
        # assign direct sample and increment the number of times it occurs
        counts[direct_sample()][1] += 1
    return counts

print compare_distributions()
