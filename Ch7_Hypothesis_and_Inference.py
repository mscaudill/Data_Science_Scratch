""" In this module we will look at Hpothesis testing and inference, we start with the example of testing whether a coin is fair based on sample flips"""

import numpy as np
import math
from Ch6_Probability import normal_cdf, inverse_normal_cdf
from matplotlib import pyplot as plt

############################################################################
# Flipping a coin n times with a probability of success on each flip is a binomial(n,p) distribution. For large n we can approximate this with a normal distribution (please see ch6 discussion of CLT)

def normal_approx_of_binomial(n,p):
    """ returns the mean mu and std sigma of the binomial(n,p)"""
    mu  = n*p
    sigma = math.sqrt(n*p*(1-p))
    return mu, sigma

############################################################################
# Total Probability below, above and between thresholds
############################################################################

# The total probability that P(Z < z) is given by the CDF
normal_probability_below = normal_cdf

# the Probability of P(Z > z) is just 1 - P(Z < z)
def normal_probability_above( lo, mu=0, sigma=1):
    return 1 - normal_cdf(lo, mu, sigma)

# The probability that P(z1 < Z < z2) 
def normal_probability_between(lo, hi, mu = 0, sigma = 1):
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# Further, we can define P that Z is not between z1 and z2
def normal_probability_outside(lo, hi, mu = 0, sigma = 1):
    return 1 - normal_probability_between(lo , hi, mu, sigmai)

############################################################################
# Return thresholds that meet probability criteria
############################################################################
def normal_upper_bound(probability, mu=0, sigma=1):
    """ returns the z for which P(Z < z) = probability """
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability, mu=0, sigma=1):
    """ returns the z for which P(Z > z) = probability """
    return  inverse_normal_cdf(1-probability, mu, sigma)

def normal_two_sided_bounds(probability, mu=0, sigma=1):
    """ returns the symmetric bounds z1,z2 where P(z1 < Z < z2)"""
    tail_probability = (1-probability)/float(2)

    # the upper bound is the lower bound of the tail probability
    upper_bound = normal_lower_bound(tail_probability, mu , sigma)

    # the lower bound is the upper bound of the tail probability
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

# Perform 1000 flips of fair coin
mu_0, sigma_0 = normal_approx_of_binomial(1000,0.5)
print mu_0, sigma_0

#########################
# Type I Err Test Signif.
#########################
# A type I error is when we reject h0 (null) even though it is true
# The two sided test provides the bounds for which if the number of heads falls outside of we reject the null hypothesis here we use a 95% confidence interval
bound_low, bound_high = normal_two_sided_bounds(0.95, mu_0, sigma_0)
print bound_low, bound_high
# This means that we reject h0 even though its true 5% of the time

#########################
# Type II error Test Power
######################### 

#A type II error looks at how often we accept H0 even though it is false this refelects the power of the test
# 95% bounds based on assumption p = 0.5
lo,hi = normal_two_sided_bounds(0.95,mu_0,sigma_0)

# actual mu sigma based on p=0.55
mu_1, sigma_1 = normal_approx_of_binomial(1000,0.55)

#type II error means we accept null when it is false. This occurs when z is withhin our null H0 interval even though it is false
type_II_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
# The power is 1 - typeIIProbability
power = 1-type_II_probability
print "power is: %f"  %(power
)
#########################
# P values
#########################
# The p-value is the probability of making a type I error given an observed value. Lets say we flip a coin 1000 times and observe 530 heads. Our null hypothese is that the coin is fair. Can we reject this null hypothesis?
def two_sided_p_value(x, mu=0, sigma=1):
    if x > mu:
        # if abs(x) is greater than the mean, note 2 comes from symm pdf
        return 2*normal_probability_above(x, mu, sigma_0)
print "The p value is: %f" %(two_sided_p_value(529.5,mu_0,sigma_0))
# The p-value is 0.062 which means there is a 6.2% change that you misidentify the coin as unfair when in fact it is fair.  Smaller p-values mean that the misidentification (type I err) is less likely and you can be more confident in rejecting the null

#########################
# Confidence Intervals
#########################
# Consider an example where we flip a coin 1000 times and observe 525 heads. We might estimate p = 525. How confident can we be about this estimate. By the CLT the number of heads should be approx normal with mean np and std = sqrt(np(1-p)). Here we don't know p so we will use our sampling estimate


p_hat = 525/float(1000)
mu = p_hat
sigma = math.sqrt(p_hat*(1-p_hat)/float(1000))

print "the confidence interval is: [%f, %f]" %(normal_two_sided_bounds(0.95,mu,sigma))

#########################
# A/B Testing
#########################
# Lets say you show two ads A and B to 1000 viewers n_a users out of N_a total click ad A n_a/N_a is a random variable with mean p_a and because these are indpt bernoulli trials std = sqrt(p_a*(1-p_a)/N_a) similarly for ad B. What we would like to know is if there is a sigificant difference in the number of choices for ad A vs B.
def estimated_parameters(N,n):
    p = n / float(N)
    sigma = math.sqrt(p*(1-p)/float(N))
    return p, sigma
# The A and B normals are indpt because they are based off indpt bernoulli trials there difference should be normal with mean p_b-p_a and std sqrt(sigma_a^2+sigma_b^2). Note here we should really use a t-distribution rather than a normal b/c the sigma is the sample std not the distributions sigma. We can now test if p_a and p_b are the same using the following statistic.
def a_b_test_statistic(N_a, n_a, N_b, n_b):
    p_a, sigma_a = estimated_parameters(N_a, n_a)
    p_b, sigma_b = estimated_parameters(N_b, n_b)
    return (p_b-p_a) / math.sqrt(sigma_a**2 + sigma_b**2) # ~ std normal
# Lets say for example A gets 200 clicks and B gets 180 clicks out of 1000. The statistic would be:
z = a_b_test_statistic(1000, 200, 1000, 180)
print "The A/B Statistic is %f" %(z)

# and the p-value for this observed statistic is
two_sided_p_value(z) #0.254 this is large meaning our probability of making a type I error is large and so we cant reject the null hypothesis (that Ads A and B are equally efective).

# if we measured 200 clicks for A and only 150 clicks for B we would get a p value of .003 meaning we can reject the hypothesis that the ads are equally effective becasue the probability of type I error is very small. Or put another way there is only a .003 probability Ads A and B are equally effective given the observed number of clicks.

##############################
# Bayesian Inference
##############################
# In Bayesian Inference we treat our unknown parameter theta as a random variable. The estimate of this parameter is updated by some prior knowledge (D) of the parameter called the prior distribution. We use Baye's Rule to get a posterior distribution f(theta|D)  which contains all the information about theta. f(theta|D) = P(D|theta)*f(theta)/P(D). Usually we will need to estimate what the prior distribution is to get our posterior estimate. 
# A comon prior distribution is the Beta distribution. Since I've never used it before we are going to explore it here. The Beta distribution has two shape parameters that alters the distribution of probabilities (alwawys bw 0 and 1)
def B(alpha, beta):
    """ a normalization constant so total probability is 1 """
    return math.gamma(alpha)*math.gamma(beta)/math.gamma(alpha + beta)

def beta_pdf(x, alpha, beta):
    if x < 0 or x > 1:
        return 0
    return x**(alpha-1)*(1-x)**(beta-1)/float(B(alpha,beta))

# plot some of the Beta PDFS
xs = [i/100.0 for i in range(0,100)]
plt.figure(4)
plt.plot(xs,[beta_pdf(x,1,1) for x in xs],'-',label='1, 1')
plt.plot(xs,[beta_pdf(x,10,10) for x in xs],'-',label='10, 10')
plt.plot(xs,[beta_pdf(x,4,16) for x in xs],'-',label='4, 16')
plt.plot(xs,[beta_pdf(x,16,4) for x in xs],'-',label='16, 4')
plt.legend(loc=9)
plt.show()
