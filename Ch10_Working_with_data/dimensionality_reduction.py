""" 
dimensionality_reduction.py

In this module, we will look at the use of PCA on two-dimensional data
where the majority of the variance lies along y = x with an offset in both x
and y.

"""

import random

from matplotlib import pyplot as plt
from functools import partial

from DS_Scratch.Ch6_Probability import inverse_normal_cdf
from DS_Scratch import Ch4_Linear_Algebra as Ch4
from DS_Scratch.Ch10_Working_with_data.rescaling import scale
from DS_Scratch import Ch8_Gradient_Descent as Ch8

# Create some normally distributed data to examine #
####################################################
def random_normal():
    """ returns a rand draw from normally distributed data """
    return inverse_normal_cdf(random.random())

xs = [random_normal() + 10 for _ in range(1000)]
ys = [x + random_normal()/2  for x in xs]

# make a matrix of the data
data = [list(tup) for tup in  zip(xs,ys)]

# Remove the mean of the data #
###############################
def de_mean(A):
    """ returns result of subtracting from every value of A the mean value
        of its column """
    num_rows, num_cols = Ch4.shape(A)
    col_means, _ = scale(A)

    return Ch4.make_matrix(num_rows,num_cols,
                       lambda i,j: A[i][j]-col_means[j])

de_meaned_data = de_mean(data)
de_xs = Ch4.get_col(de_meaned_data,0)
de_ys = Ch4.get_col(de_meaned_data,1)

# Compute Directional Variance Gradient #
#########################################

# given a directional vector 'd' (magnitude=1) each row in 'r' data extends
# in 'd' direction by dot(r,d). Lets first define a directional vector
def direction(w):
    """ returns a normalized vector in direction w """
    mag = Ch4.magnitude(w)
    return [w_i/float(mag) for w_i in w]

def directional_variance_i(r_i, w):
    """" computes for row r_i the projection of r_i onto w """
    # we square this because we later sum to get a resultant vector
    return Ch4.dot_product(r_i, direction(w))**2

def directional_variance(data,w):
    """ computes the directional variance for all rows in data """
    return sum([directional_variance_i(r_i, w) for r_i in data])

# we do not yet have the variance vector w so lets get that. We will get
# this by maximinzing the directional variance defined above. So we employ
# gradient descent, we need to first get the gradient.

def directional_variance_gradient_i(r_i, w):
    """ computes the gradient of the directioanl variance for row_i in
        data """
    return [2 * Ch4.dot_product(r_i, direction(w)) * r_ij for r_ij in r_i]

def directional_variance_gradient(data, w):
    """ computes the variance gradient for all rows in data """
    return Ch4.vector_sum([directional_variance_gradient_i(r_i, 
                            w) for r_i in data])

# Determine Principal Component by Grade. Descent #
###################################################
# PC1 is the direction that maximizes the directional_variance_gradient
def first_principal_component(data):
    guess = data[0]
    # use partial to make the target and grade fncs a variable of w only
    unscaled_maximizer, _, _ = Ch8.maximize_batch(
                            partial(directional_variance, data),
                            partial(directional_variance_gradient, data),
                            guess,tolerance = 0.00001)

    return direction(unscaled_maximizer)

principal_1 = first_principal_component(de_meaned_data)

if __name__ == '__main__':
    # plt the non-zero mean data
    plt.figure(1)
    plt.scatter(xs,ys,marker='.',color = 'black')
    plt.title('Data with non-zero mean')
    
    # plt the zero mean data
    plt.figure(2)
    plt.scatter(de_xs,de_ys, marker='.', color = 'gray')
    plt.title('data with zero mean')

    # plot the first principal component on the scaled data
    print "The principal comonent is: [%f,%f]" %(principal_1[0], 
          principal_1[1])
    ax = plt.gca()
    ax.arrow(0, 0, principal_1[0], principal_1[1], width = 0.05, 
             color = 'r', head_width=0.1, head_length=0.1, fc='r',ec='r')
    plt.show()
    
    plt.show()