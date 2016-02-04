"""
In this module we will create a set of tools for doing basic linear
algebra. It is important to remember that these tools are for teaching only
In non-toy situations you should opt fot the NUMPY library for array and
matrix operations since it has been optimized
"""

import math

############################################################################
# Vector addition
############################################################################
def vector_add(v, w):
    """ returns the vector sum of two input list v and w """
    return [x + y for x,y in zip(v,w)]

############################################################################
# Vector subtraction
############################################################################
def vector_subtract(v,w):
    """ returns the vector subtraction of list v and w """
    return[x-y for x,y in zip(v,w)]

###########################################################################
# Vector sum
############################################################################
def vector_sum(vectors):
    """ returns the vector sum of arbitrary number of vectors """
    return reduce(vector_add,vectors)

############################################################################
# Scalar Multiply
############################################################################
def scalar_multiply(c,vector):
    """ returns the scalar multiple of c (float) with a vector """
    return [c * x for x in vector]

###########################################################################
# Vector mean
############################################################################
def vector_mean(vectors):
    """ computes a mean vector from a set of vectors"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

############################################################################
# Dot product
############################################################################
def dot_product(v,w):
    """ returns the dot product of two list v,w """
    return sum([x*y for x,y in zip(v,w)])

############################################################################
# sum of squares
############################################################################
def sum_of_squares(v):
    """ returns the sum  v1**2 + v2**2 +... """
    return sum([x**2 for x in v])
    
############################################################################
# magnitude
############################################################################
def magnitude(v):
    return math.sqrt(sum_of_squares(v))

############################################################################
# squared distance
############################################################################
def squared_distance(v,w):
    """ computes the squared distance between two vectors """
    return sum_of_squares(vector_subtract(v,w))

############################################################################
# distance
############################################################################
def distance(v,w):
    """ returns the magnitude of squared distance between v and w """
    return magnitude(squared_distance(v,w))


############################################################################
# Matrix operations
############################################################################
"""A matrix using list is represented as list of list A = [[1,2,3],[4,5,6]].
So each inner list is a row of the matrix and the number of elements in each
inner list is the number of matrix columns."""

############################################################################
# matrix shape
############################################################################
def shape(A):
    """ returns number of rows and the number of cols of input matrix """
    if A: 
        num_rows = len(A)
        num_cols = len(A[0])
        return num_rows, num_cols
        # if A is empty return a shape of 0
    else: return 0

############################################################################
# extract row
############################################################################
def get_row(A,i):
    """ Gets row i of matrix A """
    return A[i]

############################################################################
# extract col
############################################################################
def get_col(A,j):
    """ gets the jth column of A """
    return [A_i[j] for A_i in A]

############################################################################
# make matrix
############################################################################
def make_matrix(num_rows,num_cols,entry_fn):
    """ makes a matrix of num_rows x num_cols whose values are given by 
        entry fn """
    return [[entry_fn(i,j) for j in range(num_cols)] for i in 
            range(num_rows)]

############################################################################
# is diagonal
############################################################################
def is_diagonal(i,j):
    """ 1's on diagonal and zeros elsewhere """
    return 1 if i==j else 0

############################################################################

