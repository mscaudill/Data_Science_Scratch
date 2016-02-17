""" 
In this module we will discuss machine learning. In particular, we will
discuss how to evaluate a model's performance in prediciting data.

What is machine learning?
Macine learning refers to the ability to learn from and make predictions
about data.

Types of models
1.  Supervised- model in which a subset of the data is labeled with correct
    answers to learn from.
2.  Unsupervised- model in which none of the data is labeled as correct
3.  Online- model which needs to adapt to newly arriving data

In most cases, we will choose a family of parameterized models and then try
to find the optimal parameters. For example we may assume a linear
relationship between two variables and then try to find the slope and
intercept that best fit the data.

Overfitting occurs when a model fits well to a training set of the data but
generalizes poorly to any new data. This usually involves models that are
too complex. So how do we ensure our models aren't too complex?
A simple way is too split your data into two sets, a training set in which
you determine the optimal parameters of the model and a testing set where
measure the model's performance with no further parameter modifications.
"""

import random

def split_data(data, prob):
    """ split data into fractions [prob, 1-prob] """
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

""" Often our data will be represented as a matrix of x input values and y
    output values so we need to make sure to put the corresponding values
    togehter in either the training or testing set """

def train_test_split(x, y, test_pct):
    """ place (x,y) tuples into training or testing data set """
    data = zip(x,y)
    train, test = split_data(data, 1-test_pct)
    xtrain, ytrain = zip(*train) # unzip trick
    xtest, ytest = zip(*test)
    return xtrain, xtest, ytrain, ytest

""" Even though a model performs well on both the train and test sets, this
does not mean the model will necessarily generalize to a larger data set. It
could be that there are still trends in the training and test sets. For
example if you were looking at user activity that appeared in both the
training and testing sets. 
A big problem is if you used the training and testing set to choose between
models. FOr example you have two models and you choose the one that performs
best on the test set. Then by definition this model performs well on the
test set but may not generalize. Instead to choose between models you should
split the data into three parts: a training set for building models, a
validation set to choose between models and a test set for juding the final
model.

Correctness
To assess a models correctness we can look at the number of type I (false
positive) and type II (false negative) errors that are being made. The
precision measures the percentage of true positive to false positives.

| tp  fp |
| fn  tn |

The recall measures the fraction of positives our model identifies
"""

def precision(tp, fp):
    """ the percentage of true-positive to false-positives i.e. the number
        we got right to the number we got wrong"""
    return tp / float(tp + fp)

def recall(tp, fn):
    """ measures the number of correct identifications to the number of
        missed identifications """
    return tp / float(tp + fn)

""" These metrics can be combined into the F1 score which is the harmonic
mean of the precision and recall metrics """

def f1_score(tp, fp, tn, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2*p*r / float(p + r)

""" Bias Variance trade-off
recall that the bias of an estimator theta is B(theta) = E[theta]-theta. For
data that is underfit no matter which training set you draw from the data
(Ex fitting a line to parabolic data). Here the bias is high. However any
two linear models will yield similar average values and so we say the
variance is low. If we fit a high degree polynomial to the parabolic data we
could have low bias since the model would fit it very well...but for any two
training sets the models would be very different and thus have high
variance. This is known as the bias 'variance trade-off' """

""" Features
features are the inputs we provide to the model and depending on the
feature(s) we will need to use different kinds of models. Example: Consider
whether an email is spam. We could use the following features

1. does the email contain the word 'Viagra' (binary 1 or 0)
2. How many times does the letter d appear (numeric)
3. What was the domain of the sender (categorical)

Naive Bayes can be used for binary data
Regression models can be used for numeric
Decision trees can be used for categorical

We will look at these in the chapters to follow.
"""





