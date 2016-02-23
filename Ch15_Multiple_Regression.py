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
import random
import math 

def predict(x_i, beta):
    """ assumes x_i1 = 1 and beta is a vector the length of x_i """
    return dot_product(x_i, beta)

""" assumptions of the multiple least squares model 
1. x_i's are independent
2. x_i's are uncorrelated with the errors epsilon """

# we need to minimize the sum of squared errors
def error(x_i, y_i, beta):
    return y_i - predict(x_i, beta)

def squared_error(x_i, y_i, beta):
    return error(x_i, y_i, beta)**2

# we will use stochastic gradient descent to minimize the squared error so
# we need to compute the squared_error gradient
def squared_error_gradient(x_i, y_i, beta):
    """ gradient with respect to beta correponding to the ith squared
        error term"""
    return [-2*x_ij*error(x_i, y_i, beta) for x_ij in x_i]

# now we are ready for stochastic gradient descent to get beta
def estimate_beta(x, y):
    # guess the initial beta
    beta_initial = [random.random() for _ in x[0]]
    return minimize_stochastic(squared_error, squared_error_gradient,
                               x, y, beta_initial, 0.001)

if __name__=='__main__':
    # Lets make some fake data to examine we will make the data of the form
    # y = 1 + 3.2*x1 - 0.5*x2 + 10*N(0,1) ; x2 = 1 / x1 and so lin. indpt
    
    constant = [1 for _ in range(19)]
    x1 = [el for el in lspace(1,10,19)]
    x2 = [1/float(el) for el in lspace(1,10,19)]
    xs = [list(tup) for tup in zip(constant, x1, x2)]

    
    ys = [xs[point][0] + 3.2 * xs[point][1] - 0.5 * xs[point][2] +
         random.random() for  point, _ in enumerate(xs)]

    plt.figure()
    plt.scatter(range(len(xs)),ys)
    plt.show()
    print estimate_beta(xs, ys)
