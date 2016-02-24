""" In this module we will look at the method of gradient descent. It is 
useful for things like MLE and MAP parameter estimation where we need to 
maximize some target function. Note that here again we will use a 
long-hand form for computing gradients etc but scipy has builtin methods 
which are much more computationally effecient."""

import math
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from Ch4_Linear_Algebra import vector_subtract, scalar_multiply
import decimal
##############################
# Create target func to minimize
##############################
"""Lets set up a target function to minimize we will make it a vector 
function meaning it takes a vector of inputs and returns a single value"""
def sum_of_squares(v):
    """ computes the sum of the squared elements in v """
    return sum([v_i**2 for v_i in v])

# make a plot of this function. This is an aside but I want to play around 
#with numpy and matplotlib some more.

#array of x values
xs = np.linspace(-10,10,100)
ys = np.linspace(-10,10,100)
xx,yy = np.meshgrid(xs,ys)
zs = xx**2 + yy**2

# See Plot Below

##############################
# Estimate Gradient
##############################
"""To estimate the gradient we will compute the partial difference quotient
Recall that for f(x) the derivative can be estimated as (f(x+h)-f(x))/h. 
This is the difference quotient. Like-wise the partial difference quotient 
is"""

def partial_difference_quotient(f, v, i, h):
    """ Computes the ith partial difference quotient of f at v"""
    # create a new vec w where h has been added to only the ith component
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)] 
    # now return the difference quotient    
    return (f(w)-f(v)) / h

# now we can estimate the gradientby calling partial difference quotient 
#on each ith component of v
def estimate_gradient(f,v,h=.0001):
    return [partial_difference_quotient(f, v, i, h) for i,_ in enumerate(v)]

"""Note the above has the major drawback in that if you have n dims in v you
need to eval f on 2*n points and if you need to estimate the gradient 
multiple times you are calculating f at values that are far from the 
gradient many times over."""

##############################
# Minimization/Maximization
##############################
"""now that we can estimate the gradient of any function lets go back to our
sum of squares ex z=x^2+y^2 and locate the minimum of this function. We 
could use our gradient estimate above but as we mentioned this is expensive 
so lets use the known gradient [2x, 2y]"""

# We need to be able to move a step from current location v in a given 
#direction with a certain stepsize

def step(v, direction, step_size):
    """ move a step_size direction from v """
    return [v_i + direction_i*step_size 
            for v_i, direction_i in zip(v,direction)]

def sum_of_squares_gradient(v):
    """ gradient of sum of squares func"""
    return [2*v_i for v_i in v]

# pick a random starting location v
v = [random.randint(-10,10) for i in range(2)]

# set a convergence tolerance (distance between v and previous v)
tolerance = .00000001
iteration_number = 0

while True:
    # Compute the gradient at v
    gradient = sum_of_squares_gradient(v)
    # compute the next v in the direction of the gradient
    next_v = step(v, gradient, -0.01)
    # compute the distance between v and next_v
    distance = sum([(next_v[i] - v[i])**2 for i,_ in enumerate(v)])
    if distance < tolerance:
        break
    v = next_v
    iteration_number += 1

minimum_z = sum_of_squares(v)

"""A potential problem here is how do we know what step size to choose? One 
way to do this is to provide a list of step sizes and test each one. However
it is possible that some step size will result in invalid inputs for our 
function so we create a safe apply function."""
def safe(f):
    """ return a new function the same as f but returns infinity when f 
        produces and error"""
    def safe_f(*args, **kwargs):
        try:
            return f(*args,**kwargs)
        except:
            return float('inf')
    return safe_f

##############################
# General gradient descent
##############################
"""In the general case, we have some target function that we would like to 
minimze and its gradient function and a set of starting parameters theta_0. 
Here is the general implementation of the gradient descent."""

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance = 0.00001):
    """ use gradient to find theta that minimizes target func """
    # set a variable step size, we'll locate the optimal
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 
                  0.00001, .000001]
    
    # set the initial value of the parameters 
    theta = theta_0
    # set the target function to be the sage version
    target_fn = safe(target_fn)
    # value to be minimized
    value = target_fn(theta)
    
    iteration = 0;

    while True:
        # compute the gradient at the current theta set
        gradient = gradient_fn(theta)
        # take a step along the gradient
        next_thetas = [step(theta, gradient,-step_size) 
                        for step_size in step_sizes]
        # choose the next_theta that minimizes the target func
        next_theta = min(next_thetas, key=target_fn)
        # update the next value of the function
        next_value = target_fn(next_theta)
        
        # stop if we are converging
        if abs(value - next_value) < tolerance:
            return theta, value, iteration

        elif iteration == 10000:
            print 'Exceeded 10000 iterations. Last theta: [%.5f, %.5f],'\
            " Last Value: %.5f" %(theta[0],theta[1], value)

            return theta, value, iteration

        else:
            theta, value = next_theta, next_value
            iteration += 1

# We also need a way to compute maximums of target_fn we will do this by 
# minimizing the negative of f
def negate(f):
    """ return a function that for any input x returns -f(x)"""
    return lambda *args, **kwargs: -f(*args, **kwargs)

def negate_all(f):
    """ retrurn the negation of a list from f"""
    return lambda *args, **kwargs: [-y for y in f(*args,**kwargs)]

# now we can define maximize_batch
def maximize_batch(target_fn, gradient_fn, theta_0, tolerance = 0.00001):
    return minimize_batch(negate(target_fn), negate_all(gradient_fn), 
                          theta_0,
                          tolerance)
                            
##############################
# Stochastic gradient descent
##############################
"""In SGD, we will choose an initial vector of parameters theta_0 and a 
step_size/learning rate eta. Then repeat the following (1) randomly shuffle 
data in the training set (2) for i=1,..n do 
theta := theta - eta*gradient(target_fn)"""

def in_random_order(data):
    """generator that returns the elements of data in random order"""
    # create a list of indexes
    indexes = [i for i,_ in enumerate(data)]
    # shuffle
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]

def minimize_stochastic(target_fn, gradient_fn, x, y, 
                        theta_0, eta_0 = 0.01):
    data = zip(x,y)
    theta = theta_0
    eta = eta_0

    # set the current minimum theta and min_val of target_fn
    min_theta, min_value = None, float('inf')

    # initialize the number of iterations with no decrease in value
    iterations_with_no_improvement = 0

    # Do not let iterations exceed 100
    while iterations_with_no_improvement < 100:
        
        #print 'iterations with no improvement: %g' %(
              #iterations_with_no_improvement)
        
        value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in data )

        if value < min_value:
            # if the new value is less than the old one save it and go
            # back to the original step size
            min_theta, min_value = theta, value
            eta = eta_0
        else:
            # we are not improving so shrink the step size and 
            # increment iteration
            iterations_with_no_improvement += 1
            # see if eta is a decimal type and match increment accordingly
            if type(eta)==decimal.Decimal:
                eta *= decimal.Decimal( 0.9)
            else:
                eta *= 0.9

        # take a gradient step for each of the data points
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(eta,gradient_i))

    return min_theta

def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, 
                        eta_0 = 0.01):
    return minimize_stochastic(negate(target_fn), negate_all(gradient_fn),
                               x, y, theta_0, eta_0)

if __name__ == '__main__':
    
    # plot of the target function to minimize
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, zs, cmap = cm.jet)
    plt.show()

    # computed by hand method
    print 'Hand Method'
    print '[x,y]= [%.5f, %.5f] yields a minimum z ' \
          'of %.5f in %d iterations'  %(v[0],v[1],
          minimum_z, iteration_number)

    # computed using minimize_batch method
    print 'Batch Method'
    theta, value, iteration = minimize_batch(sum_of_squares, 
                                              sum_of_squares_gradient, 
                                              [2,2])
    print '[x,y]= [%.5f, %.5f] yields a minimum z of %.5f in %d'\
          ' iterations' %(theta[0], theta[1], value, iteration)

