"""
In this module we will look at multiple regression. For each x_i,y_i pair we
have y_i = beta_0 + beta_1*x_i1 + beta_2*x_i2 + ... Here we can treat this
as a linear algebra problem Y = X*beta + eps where X is a matrix of
coeffecients and beta, epsi are vectors. We again will solve the problem
using minimization of sum_of_squared errors.
"""
from DS_Scratch.Ch4_Linear_Algebra import dot_product

def predict(x_i, beta):
    """ assumes x_i1 = 1 and beta is a veector the length of x_i """
    return dot_product(x_i, beta)

""" assumptions of the multiple least squares model 
1. x_i's are independent
2. x_i's are uncorrelated with the errors eps """

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
