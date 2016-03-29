"""
A small module exploring eigenvectors and eigenvalues of matrices with an
algortihm that ineffeciently computes eigenvectors by rotating a guess
vector around until it hits the eigenvector.
"""
from DS_Scratch.Ch4_Linear_Algebra import *
from functools import partial
import random

def matrix_product_entry(A, B, i, j):
    """ returns i,j component of matrix multplication of matrices A & B """
    return dot_product(get_row(A,i), get_col(B,j))

def matrix_multiply(A, B):
    """ returns the matrix multiplication of matrices A and B """
    num_rows_A, num_cols_A = shape(A)
    num_rows_B, num_cols_B = shape(B)

    if num_cols_A != num_rows_B:
        raise ArithmeticError("Input Matrix Sizes Incompatible!")

    return make_matrix(num_rows_A, num_cols_B,
                       partial(matrix_product_entry, A, B))

def vector_as_matrix(v):
    """ returns vector v (represented as a list) as a n x 1 matrix """
    return [[v_i] for v_i in v]

def vector_from_matrix(v_as_matrix):
    """ returns a vector (represented as a list) from an n x 1 matrix """
    return [row[0] for row in v_as_matrix]

def matrix_operate(A, v):
    """ multiplies matrix a by vector v """
    # represent v as matrix
    v_as_matrix = vector_as_matrix(v)
    product = matrix_multiply(A, v_as_matrix)
    return vector_from_matrix(product)

# Finding Eigenvectors and Eigenvalues #
########################################
# One way to manually find eigenvectors is by picking a starting vector v,
# transform it under A, rescale the result to have magnitude 1 and repeat
# the process. The idea here is that each matrix multiplication is simply
# rotating and scaling the vector v, so if we repeatedly do this we should
# rotate v around till we get the eigenvector. It's a silly method given
# that we can compute it directly but lets try it out.

def find_eigenvector(A, tolerance=0.00001):
    guess = [random.random() for _ in A]
    iteration = 0
    
    while True and iteration < 10000:
        result = matrix_operate(A, guess)
        length = magnitude(result)
        next_guess = scalar_multiply(1/float(length), result)

        if distance(guess, next_guess) < tolerance:
            # return the eigenvector and eigenvalue
            return next_guess, length
        else:
            guess = next_guess
            iteration += 1

if __name__ == '__main__':
    # Eigenvector test: compute using algorithm
    M = [[3,4],[2,1]]
    # compute the eigenvector, eigenvalue
    eigenvector, eigenvalue = find_eigenvector(M)
    print "(Eigenevector, Eigenvalue)"
    print eigenvector, eigenvalue
    
    print "Scalar Multiplication of Eigenvector by Eigenvalue"
    print scalar_multiply(eigenvalue,eigenvector)

    # check if matrix_operate yields back eigenvector
    print "Matrix multiplication yields"
    print matrix_operate(M,eigenvector)
     
