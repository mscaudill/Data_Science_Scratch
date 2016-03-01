""" In this module we are given a set of data for data science users. The
data consist of years experience, salary and whether the user has a paid
account on a website. We would like to predict whether a user will have a
paid account based on years experience and salary. We will first try a
linear regression model (paid = beta0 + beta_1*experience + beta_2*salary.
This model will fail. We then will use a logistic regression model """

from sample_data import data as data
from matplotlib import pyplot as plt
from DS_Scratch.Ch10_Working_with_data.rescaling import norm_matrix
from DS_Scratch.Ch15_Multiple_Regression import estimate_beta, predict

# Format data #
###############
# data is a list of tuples (years_experience, salary, paid_account) we
# format this into x = [1, years_experience, salary] and y = [paid acct].
# Note the [1] is for the constant beta_0 term in our model.
x = [[1] + list(row[:2]) for row in data]
y = [row[2] for row in data]


# Rescale data and Beta estimate #
##################################
rescaled_x = norm_matrix(x)
beta = estimate_beta(rescaled_x, y)
# compute predictions from beta estimate and rescaled x
predictions = [predict(x_i, beta) for x_i in rescaled_x]

""" In the plot, we notice several problems using this approach. First, we
are trying to predict class membership (0 or 1) for paid accounts. But some
of the outputs of the linear model are negative which is hard to interpret.
Second, the ouputs are forced to be either 0 or 1 and yet we find that the
regression coeffecient beta_1 for experience is large and positive (0.43)
so as experience goes up the error term must go down to maintain a value
less than 1. This means the cols of x are not indpt from the error term and
our value of beta is biased. What we need is for large + values to
correspond to membership 1 and negativ values to correspond to membership
0. This is where the logistic function is useful. """



if __name__ == '__main__':
    
    # Scatter years vs salaries for paid/unpaid #
    #############################################
    plt.figure(1)
    plt.scatter([row[0] for row in data if row[2]==1], 
                [row[1] for row in data if row[2]==1], marker='o',
                label = 'paid')

    plt.scatter([row[0] for row in data if row[2]==0], 
                [row[1] for row in data if row[2]==0], marker='*', 
                color='r', label = 'unpaid')
    plt.legend(loc = 'lower center')

    # Plot Linear Regression Predictions and Actuals #
    ##################################################
    plt.figure(2)
    # plot the predicted paid accounts and the actual paid accounts
    plt.scatter(predictions, y)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    
    plt.show()
