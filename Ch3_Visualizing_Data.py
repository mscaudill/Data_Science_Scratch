"""
This module provides examples of basic plotting with matplotlib libraries
"""


# Perform imports of needed modules
from matplotlib import pyplot as plt
import random 
from collections import Counter
import string

############################################################################
# Nominal GDP Plot
############################################################################
# Create some ficticious data
years = [1950+x for x in [10*y for y in range(0,7)]]
gdp = [100*random.random() for _ in range(0,7)]

# Create a line chart
plt.figure(1)
plt.plot(years,gdp,color = 'green', marker = 'o', linestyle = 'solid')
# Add a title
plt.title('Nominal GDP')
plt.ylabel('Billions of $')


############################################################################
# Bar PLots: Grades Ex 
############################################################################
# Create some ficticious grades
grades = [70*random.random() for _ in range(0,19)]

# Write a function that rounds each grade to the nearest ten
def decile(grade):
    """ Rounds a grade to the nearest ten """
    return round(grade/10.0)*10

# Construct a histogram using the counter dict obj
histogram = Counter(decile(grade) for grade in grades)

# Create a simple bar plot
plt.figure(2)
# Center each decile by shifting 4 and show each bar with width 8
plt.bar([x-4 for x in histogram.keys()],histogram.values(),8)

# Set the plot axis
plt.axis([round(min(grades))-5,round(max(grades))+5, 0,
          max(histogram.values())])

# plot x ticks
plt.xticks([10*i for i in range(11)]) # grades go from 0:100
plt.xlabel('Decile')
plt.ylabel('# Students')
plt.title ('Grade Distribution')

############################################################################# Multiple Line Charts
############################################################################# Here we demonstrate how to construct line charts with multiple plots
# make some fake data to plot
variance = [2**i for i in range(1,10)]
bias_squared = variance[::-1]

total_error = [x + y for x, y in zip(variance, bias_squared)]

xs = [i for i, _ in enumerate(variance)]

# make our plots
plt.figure(3)
plt.plot(xs, variance, 'g-', label = 'variance')
plt.plot(xs, bias_squared, 'r-.', label = 'bias^2')
plt.plot(xs, total_error, 'b:', label = 'total error')

# we have labels so a legend can be applied
plt.legend(loc = 9) # loc = 9 means top center
plt.xlabel('Model Complexity')
plt.title('The Bias-Variance Tradeoff')

############################################################################
# Scatter plots
############################################################################
# Make some fake data
friends = [random.randrange(60,71) for _ in range(0,10)]
minutes = [random.randrange(100,250) for _ in range(0,10)]
# get the first 10 letters of the alphabet using string.letters
labels = list(sorted(set(string.letters.lower())))[0:10]

# make the plot
plt.figure(4)
plt.scatter(friends, minutes)
# now label each point
for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label, xy = (friend_count, minute_count), 
                 xytext = (5,-5), textcoords='offset points')
plt.xlabel('# friends')
plt.ylabel('Minutes spent at website/day')
plt.title('Minutes at site vs. number of friends')
plt.show()
