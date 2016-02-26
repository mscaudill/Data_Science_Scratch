"""
In this module we will look at multiple regression. For each x_i,y_i pair we
have y_i = beta_0 + beta_1*x_i1 + beta_2*x_i2 + ... Here we can treat this
as a linear algebra problem Y = X*beta + eps where X is a matrix of
coeffecients and beta, epsi are vectors. We again will solve the problem
using minimization of sum_of_squared errors.
"""
from DS_Scratch.Ch4_Linear_Algebra import dot_product
from Ch8_Gradient_Descent import minimize_stochastic
from matplotlib import pyplot as plt
from numpy import linspace as lspace
from tqdm import tqdm
import random
import math 
import decimal

# Multiple linear regression: Prediction #
##########################################
def predict(x_i, beta):
    """ assumes x_i1 = 1 and beta is a vector the length of x_i """
    return dot_product(x_i, beta)

""" assumptions of the multiple least squares model 
1. x_i's are independent
2. x_i's are uncorrelated with the errors epsilon """

# Multiple linear regression SE objective function #
#####################################################
def error(x_i, y_i, beta):
    return y_i - predict(x_i, beta)

def squared_error(x_i, y_i, beta):
    return error(x_i, y_i, beta)**2

# SGD Minimization of SE #
##########################
# we will use stochastic gradient descent to minimize the squared error so
# we need to compute the squared_error gradient
def squared_error_gradient(x_i, y_i, beta):
    """ gradient with respect to beta correponding to the ith squared
        error term"""
    return [-2*x_ij*error(x_i, y_i, beta) for x_ij in x_i]

# now we are ready for stochastic gradient descent to get beta
def estimate_beta(x, y):
    # guess the initial beta
    beta_initial = [decimal.Decimal(random.random()) for _ in x[0]]
    return minimize_stochastic(squared_error, squared_error_gradient,
                               x, y, beta_initial, decimal.Decimal(0.001))

# Goodness of fit #
###################
def multiple_r_squared(x, y, beta):
    """ computes the r-squared value between the predicted y and the actual
        y"""
    sum_of_squared_errors = sum(error(x_i, y_i, beta)**2 
                                for x_i, y_i in zip(x, y))
    
    return 1 - sum_of_squared_errors / float(variance(y))

# Bootstrapping #
#################
def bootstrap_sample(data):
    """ randomly samples len(data) elements with replacement """
    return [random.choice(data) for _ in data]

def bootstrap_statistic(data, stats_fn, num_samples):
    """ evaluates stats_fn on num_samples bootstrap samples from data) """
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]

# Standard Errors of Regression Coeffecients #
##############################################
""" The standard deviation of the beta estimate in our model is what we are
after here. This error can be computed directly using multivariate
statistics. I've ordered a book to help with this. We can also use the
bootstrap method to estimate these errors. We do that here. """

def estimate_sample_beta(sample):
    """ sample is a list of pairs i(x_i, y_i). We do this becasue the random
        choice must sample x_i with the corresponding y_i """
    # unpack the sample    
    x_sample, y_sample = zip(*sample)
    # return beta estimate for this sample
    return estimate_beta(x_sample, y_sample)


if __name__=='__main__':
    # Lets make some fake data to examine we will make the data of the form
    # y = 1 + 3.2*x1 - 0.5*x2 + 10*N(0,1) ; x2 = x**2 and so lin. indpt
    # Important note: when I first tried to find the beta = [1, 3.2, 0.5] I
    # used floats to represent numbers but floats are double precision and
    # cause an overflow. So I represent numbers here as decimals with 8
    # precision.
    
    decimal.getcontext().prec = 8

    constant = [decimal.Decimal(1) for _ in range(99)]
    x1 = [decimal.Decimal(el) for el in lspace(-4,4,99)]
    x2 = [decimal.Decimal(el**2) for el in lspace(-4,4,99)]
    xs = [list(tup) for tup in zip(constant, x1, x2)]

    
    ys = [xs[point][0] + decimal.Decimal(3.2) * xs[point][1] -
          decimal.Decimal(0.5) * xs[point][2] + 
          decimal.Decimal(random.random()) for  point, _ in enumerate(xs)]

    plt.figure()
    plt.scatter(range(len(xs)),ys)
    plt.show()
    print estimate_beta(xs, ys)

    # now get the error estimate of betas
    bootstrap_betas = bootstrap_statistic(zip(xs, ys), 
                                          estimate_sample_beta,
                                          100)

    # convert the bootstrap betas back to floats
    bootstrap_betas = [[float(beta_ls[i]) for i in range(3)] for beta_ls in
                        bootstrap_betas]
    # calculate the standard errors                    
    bootstrap_standard_errors = [standard_deviation([beta[i] for beta in 
                                 bootstrap_betas]) for i in range(3)]
    print "The Bootstrapped Standard Error is: "
    print bootstrap_standard_errors
    
