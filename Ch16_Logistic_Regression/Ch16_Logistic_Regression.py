""" In this module we are given a set of data for data science users. The
data consist of years experience, salary and whether the user has a paid
account on a website. We would like to predict whether a user will have a
paid account based on years experience and salary. We will first try a
linear regression model (paid = beta0 + beta_1*experience + beta_2*salary.
This model will fail. We then will use a logistic regression model """

from sample_data import data as data
from matplotlib import pyplot as plt
from DS_Scratch.Ch10_Working_with_data.rescaling import norm_matrix, scale
from DS_Scratch.Ch15_Multiple_Regression import estimate_beta, predict
from DS_Scratch.Ch4_Linear_Algebra import dot_product, vector_add
from DS_Scratch.Ch11_Machine_Learning import train_test_split
from DS_Scratch.Ch8_Gradient_Descent import maximize_stochastic
from DS_Scratch.Ch5_Data_Statistics import standard_deviation
from functools import partial
from numpy import linspace

import random
import math

# Format data #
###############
# data is a list of tuples (years_experience, salary, paid_account) we
# format this into x = [1, years_experience, salary] and y = [paid acct].
# Note the [1] is for the constant beta_0 term in our model.
x = [[1] + list(row[:2]) for row in data]
y = [row[2] for row in data]


# Rescale data and Linear Model Beta estimate #
###############################################
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

# Logistic function #
#####################

def logistic(x):
    return 1.0/(1 + math.exp(-x))

""" The PDF of our model y_i = f(x_i*beta)+e_i can be written as
p(y_i;x_i,beta) = f(x_i*beta)^y_i * (1-f(x_i*beta))^(1-y_i). Our goal is to
find beta that maximizes the likelihood function which is just a product of
p(beta; x_i, y_i) above. We will actually maximize the log of the
likelihood. The log likelihood is log(L(beta; x_i,y_i)) =
y_i*log(f(x_i*beta))+(1-y_i)*log(1-f(x_i*beta)). We do all of this instead
of maximizing the squared error because in the case of a logistic funct f,
the beta that maximizes the squared error is not the same as the beta that
maximizes the likelihood function"""

def logistic_log_likelihood_i(x_i, y_i, beta):
    """ returns the log likelihood of the ith bernoulli trial """
    if y_i == 1:
        return math.log(logistic(dot_product(x_i,beta)))
    else:
        return math.log(1-logistic(dot_product(x_i,beta)))

""" The total likelihood is just the product of the probabilities (indpt
assumption) and therefore the log of the likelihood is just the sum of the
individual log likelihoods"""

def logistic_log_likelihood(x, y, beta):
    """ this will be a function of beta to be maximized """
    return sum(logistic_log_likelihood(x_i,y_i,beta) 
               for x_i,y_i in zip(x,y))

""" To get beta we are going to need the gradient of the logistic log
likelihood. This I have worked out on paper (see notes) as """
def logistic_log_partial_ij(x_i, y_i, beta, j):
    """ here i is the point index and j is the gradient index """
    return (y_i - logistic(dot_product(x_i,beta))) * x_i[j]

""" now we need to get the full gradient at point x_i,y_i by summing all the partials """
def logistic_log_gradient_i(x_i, y_i, beta):
    """ returns the gradient of the log likelihood for point i """
    return [logistic_log_partial_ij(x_i, y_i, beta, j) for j,_ in
            enumerate(beta)]
""" finally we get the gradient at all points """
def logistic_log_gradient(x, y, beta):
    """ returns the full logistic log gradient across all points """
    return reduce(vector_add, [logistic_log_gradient_i(x_i, y_i, beta) 
                  for x_i, y_i in zip(x,y)])

# Apply the model #
###################
random.seed(0)
# Construct training and test sets
x_train, x_test, y_train, y_test = train_test_split(rescaled_x, y, 0.33)

# get likelihood and gradient likelihood functions as a func of beta only
target_fn = partial(logistic_log_likelihood, x_train, y_train)
gradient_fn = partial(logistic_log_gradient, x_train, y_train)

 # see main for application

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

    # Compute beta_hat by SGD #
    ###########################
    # pick a random initial beta (constant, beta1*experience, beta2*salary)
    beta_0 = [random.random() for _ in range(3)]
    beta_hat = maximize_stochastic(logistic_log_likelihood_i,
                                   logistic_log_gradient_i,
                                   x_train, y_train, beta_0)
   
    print "beta_hat", beta_hat

    # Transform beta_hat back to unscaled variables #
    #################################################
    # get the means and stds of const, years, experience cols in data
    means_x, stds_x = scale(x)

    # beta_i i!=0 has the following transform beta_i = beta_i_scaled/sigma_i
    # and beta_0 is 
    beta_hat_unscaled =[beta_hat[0],
                        beta_hat[1]/stds_x[1], 
                        beta_hat[2]/stds_x[2]]
    print "beta_hat_unscaled", beta_hat_unscaled
    
    # Fit Quality #
    ###############
    # Examine the test data
    true_positives = false_positives = true_negatives = false_negatives = 0

    for x_i, y_i in zip(x_test, y_test):
        # For the test data get a prediction for y. This will be a
        # probability between 0 and 1
        predict = logistic(dot_product(beta_hat, x_i))

        # Set a threshold of 0.5 to make our prediction a binary 0 or 1
        if y_i == 1 and predict >= 0.5:
            # increment true positives
            true_positives += 1
        elif y_i == 1:
            # increment false negatives
            false_negatives += 1
        elif predict >= 0.5:
            # increment false positives
            false_positives += 1
        else:
            true_negatives += 1
    
    # Get the precision and recall
    precision = true_positives/float(true_positives + false_positives)
    recall = true_positives/float(true_positives + false_negatives)
    print "Prediction Precision = %.3f" %(precision)
    print "Prediction Recall = %.3f" %(recall)

    # Hyperplane #
    ##############
    """ The set of points where dot_product(beta_hat*x_i) = 0 is a decision
    boundary. Here logistic(0) = 0.5 which we used to separate a paid from
    unpaid account. This set of points is called a hyperplane. We got this
    as a consequence of using the logistic model. It is possible to
    calculate directly a hyperplane without calculating maximizing the 
    likelihood function of a model. This is not treated in this book. It
    comes under the heading of Support Vector Machines"""
   

    years_experience = [row[1] for row in x]

    years = linspace(min(years_experience),
                                max(years_experience),100)

    hyperplane_ys = [(7.61 + 1.42* year)/0.000240 
                      for year in years]
    
    plt.figure(1)
    plt.plot(years, hyperplane_ys)
    plt.title("Logistic Regression Decision Boundary")



    plt.show()
