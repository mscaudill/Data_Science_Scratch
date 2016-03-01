"""
rescaling.py

Many techniques for analyzing data are sensitive to the scale of data so we
need to normalize data. For a rand variable X the normalized version is
(X-bar(X))/std(X). Here we have a few functions to accoplish this for a data
matrix

"""

from DS_Scratch.Ch4_Linear_Algebra import shape, get_col, make_matrix
from DS_Scratch.Ch5_Data_Statistics import mean, standard_deviation

def scale(data_matrix):
    """ returns the means and stds of each column in data_matrix """
    num_rows, num_cols = shape(data_matrix)

    means = [mean(get_col(data_matrix, col)) for col in range(num_cols)]
    stds = [standard_deviation(get_col(data_matrix, col)) for 
            col in range(num_cols)]
    return means, stds

def norm_matrix(data_matrix):
    """ rescales the input columns to have zero mean and stdev 1. Ignores
        cols with no deviation """
    means, stds = scale(data_matrix)

    def norm(i,j):
        if stds[j] > 0:
            return (data_matrix[i][j] -means[j])/ stds[j]
        else:
            return data_matrix[i][j]

    
    num_rows, num_cols = shape(data_matrix)
    return make_matrix(num_rows,num_cols, norm)
