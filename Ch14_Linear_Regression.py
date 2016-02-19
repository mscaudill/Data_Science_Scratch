"""
Simple Linear Regression performs a linear regression analysis on x,y data
pairs. In particular we will find beta_0 and beta_1 such that y^ = beta_0 +
beta_1*x + eps_i is the line of best fit to the data. We assume eps_i is
N(0,sigma^2). The line of best fit is the line that minimizes the sum of the
squared errors. We will calculate this line two ways.
"""
from DS_Scratch.Ch5_Data_Statistics import mean, variance, covariance
from matplotlib import pyplot as plt
import random

# assuming we have a beta_0 and beta_1 we predict y_i with:
def prediction(beta_0, beta_1, x_i):
    return beta_0 + beta_1*x_i

# and the error of this prediction for y_i is
def error(beta_0, beta_1, x_i, y_i):
    """ error is the difference between y_i and the predicted value """
    return y_i - prediction(beta_0, beta_1, x_i)


# the beta_0 and beta_1 that fits the data will minimize the squared errors
def sum_of_squared_errors(beta_0, beta_1, x, y):
    return sum(error(beta_0, beta_1, x_i, y_i )**2 for x_i, y_i in zip(x,y))

# Method I: the value of beta_0 and beta_1 can be calculated directly.
# beta_0 = EY-beta_1*EX (from taking the expectation of y=beta_0+beta_1*x
# and beta_1 = cov(X,Y)/var(X) (from taking the cov(x,y=beta_0+beta_1*x))
def least_squares_fit(x, y):
    """ given training values for x,y computes the least squares values for
        beta_0 and beta_1"""
    beta_1 = covariance(x,y)/variance(x)
    beta_0 = mean(y) - beta_1 * mean(x)
    return beta_0, beta_1

# We also need to report the amount of variance that is accounted for by X
# and epsilon. Recall this is just the squared correlation coeffecient r**2
# called the coeffecient of determination. In the book the author does this
# differently recall var(Y) = beta_1**2 * var(x) + var(eps). If we divide by
# var(y) the first term on the RHS will be r**2 and the second will be
# var(eps)/var(Y). The author calculates r**2 as 1-var(eps)/var(Y). We
# calculate the first term directly here
def r_squared(x, y):
    """ Return the coeffecient of determination (the squared correlation) of
    the x, y pairs """
    return (covariance(x,y))**2 / float(variance(x)*variance(y))


if __name__ == '__main__':

    # make some fake data to play with
    x = [random.random() + 5 for _ in range(100)]
    y = [3*x_i + random.random()/2 for x_i in x]

    plt.scatter(x,y)

    beta_0, beta_1 = least_squares_fit(x,y)

    plt.plot(x,[beta_0 + beta_1*x_i for x_i in x])
    plt.show()
    print "r^2 is: %.3f" %(r_squared(x,y))
