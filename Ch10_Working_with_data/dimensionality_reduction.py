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
    guess = [1 for _ in data[0]]
    # use partial to make the target and grade fncs a variable of w only
    unscaled_maximizer, _, _ = Ch8.maximize_batch(
                            partial(directional_variance, data),
                            partial(directional_variance_gradient, data),
                            guess,tolerance = 0.00001)

    return direction(unscaled_maximizer)

principal_1 = first_principal_component(de_meaned_data)

# Determine Principal Component by SGD #
########################################
def first_principal_component_SGD(data):
    """ Uses SGD method to determine the direction that maximizes the
        directional_variance gradient """
    guess = data[0]
    unscaled_maximizer, _ = Ch8.maximize_stochastic(lambda x,_,w:
                            directional_variance_i(x,w), lambda x,_,w:
                            directional_variance_gradient_i(x,w), data, 
                            [None for _ in data], guess)
    return direction(unscaled_maximizer)

principal_1_sgd = first_principal_component_SGD(data)

# Project data onto PC #
########################
def project(v,w):
    """ projects vector v onto w """
    projection_length = Ch4.dot_product(v,w)
    return Ch4.scalar_multiply(projection_length,w)

# Remove PC to find more components #
#####################################
def remove_projection_from_vector(v,w):
    """ projects v onto w and then subtracts the projection from v """
    return Ch4.vector_subtract(v, project(v,w))

def remove_projection(data,w):
    return [remove_projection_from_vector(r_i,w) for r_i in data]

# Functions for higher order data sets #
########################################
# On higher dimensional data sets we can iteraviely find PCs remove them and
# find the next one. Here are the functions to accomplish this
def principal_component_analysis(data, num_components):
    """ employs gradient descent to compute the principal components of
        data """
    components = []
    for _ in range(num_components):
        # compute principal component
        component = first_principal_component(data)
        components.append(component)
        data = remove_projection(data, component)
    return components

def transform_vector(v, components):
    """ projects a data vector (row) onto the components """
    return [dot(v, w) for w in components]

def transform(data, components):
    """ removes the principal components of data returning a lower dim data
        set """
    return [transform_vector(r_i, components) for r_i in data]


# If run as Main #
##################
if __name__ == '__main__':
    # plt the non-zero mean data
    plt.figure(1)
    plt.scatter(xs,ys,marker='.',color = 'black')
    plt.title('Data with non-zero mean')
    
    # plt the zero mean data
    plt.figure(2)
    plt.scatter(de_xs,de_ys, marker='.', color = 'gray')

    # plot the first principal component on the scaled data
    print "The principal comonent is: [%f,%f]" %(principal_1[0], 
          principal_1[1])
    print "The principal comonent by SGD is: [%f,%f]" %(principal_1_sgd[0],
          principal_1_sgd[1])

    ax = plt.gca()
    ax.arrow(0, 0, principal_1[0], principal_1[1], width = 0.01, 
             color = 'r', head_width=0.05, head_length=0.1, fc='r',ec='r')
    plt.title('Zero-meaned data with first Principal Component')
    
    # Plot the data projected onto the first PC
    plt.figure(3)
    projected_data = remove_projection(de_meaned_data,principal_1)
    de_x_proj = Ch4.get_col(projected_data,0)
    de_y_proj = Ch4.get_col(projected_data,1)

    plt.scatter(de_x_proj,de_y_proj,marker='.',color='blue')
    
    
    plt.show()
