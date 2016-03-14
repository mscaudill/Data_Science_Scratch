""" In this module we will create two neural networks. The first will
perform an exclusive or operation XOR and the second will read in handrawn
digits and attempt to predict the actual number. We will use a technique
called backpropagation to adjust the weights that a hidden layer applies to
each of it's inputs and the output layer weights. This is the same as
minimizing the error between the output and targets using SGD. 
"""

import math
import random
from matplotlib import pyplot as plt
from matplotlib import cm
from DS_Scratch.Ch4_Linear_Algebra import dot_product
from DS_Scratch.Ch18_Neural_Networks.raw_data_digits import raw_digits
from tqdm import tqdm

# And/Or/Not Perceptrons #
##########################
""" a perceptron approximates a single neuron. It recieves n binary inputs
and responds if the weight sum of the inputs is greater than 0"""

def step_function(x):
    """ returns 1 if input > 0 """
    return 1 if x >=0 else 0

def perceptron(weights, bias, x):
    """ returns 1 if the weighted sum of x exceeds 0 and 0 if not """
    output = dot_product(weights, x) + bias
    return step_function(output)

# weights = [2,2] and bias = -3 is an AND
# weights = [2,2] and bias = -1 is an OR
# weights = [-2], and bias = 1 is a NOT

# Feed-Forward Neural Networks #
################################
# Input Layer -> Hidden Layer -> Output Layer
def sigmoid(t):
    return 1 / float(1 + math.exp(-t))

def neuron_output(weights, inputs):
    """ returns the smoothed weighted sum of inputs. Here len(weights) is 1
        greater than inputs to hold a spot for the bias term """
    return sigmoid(dot_product(weights, inputs))

def feed_forward(neural_network, input_vector):
    """ takes in a neural network as a list (layers) of a list (neurons) of
        a list (weights) and returns the output of feedforward propagation 
        of the input. """
    
    outputs = []

    # we process one layer at a time
    for layer in neural_network:
        # For simplification we will use a constant bias of 1
        input_with_bias = input_vector + [1]
        # output will be a list of nums, one for each neuron in this layer
        output = [neuron_output(neuron, input_with_bias) 
                  for neuron in layer]
        # outputs will be a list (layers) of list of nums (neurons)
        outputs.append(output)

        # now new input to the next layer is the output we just calculated
        input_vector = output

    return outputs

# Train network by Backpropagation #
####################################
# In general we won't know the weights of the neurons in the network so we
# will tune each neuron in the network on a training set so that the error 
# function i.e. the outputs of the network match some targets we provide
def backpropagate(network, input_vector, targets):
    """ tunes the weights in network so that erro between network output and
        targets is minimized. """
    # get the outputs of the network layers
    hidden_outputs, outputs = feed_forward(network, input_vector)
    
    # compute the deltas for the output layer. note deriv of logistic is
    # logistic * (1 - logisitic)
    output_deltas = [output * (1 - output) * (output-target)
                     for output, target in zip(outputs, targets)]
    
    # update the weights(j) of each of the output neurons(i)
    for i, output_neuron in enumerate(network[-1]):
        # For each output neuron (i) and response (network[-1]), we will get
        # the inputs (i.e. the hidden outputs; j of them) and modify the jth
        # weight of the ith output neuron along the gradient
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output

    # back-propagate the errors to the hidden layer. This is basically the
    # same as above but in reverse we compute the gradient as o(1-o) times
    # the input which now comes from the output layer for each neuron in the
    # hidden layer
    hidden_deltas = [hidden_output * (1 - hidden_output) * 
                     dot_product(output_deltas, 
                                 [n[i] for n in output_layer])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # update the weights (j) of the hidden layer neurons (i)
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input

    """ A much cleaner way of doing this is to compute the gradient of the
    squared error of the outputs and inputs. Please see my personal notes
    for the calculus and pseudo-code that accomplishes this. """

if __name__ == '__main__':

    # XOR Gate (Exclusive OR) #
    ###########################
    # Now we can build an exclusive or gate by combining an AND with an OR 
    # we simply negate the 'and' input so the output only reponds to OR
    xor_network = [# hidden layer
                   [[20, 20, -30], # AND neuron
                   [20, 20, -10]], # OR neuron
                   # output layer
                   [[-60, 60, -30]]] # a 2nd but not 1st responding neuron
    
    print 'XOR NETWORK---------------------------'
    for input_1 in [0,1]:
        for input_2 in [0,1]:
            # feed forward returns the output of each neuron in each layer.
            # The last one is the output neuron hence [-1]
            print input_1, input_2, feed_forward(xor_network,
                                                 [input_1, input_2])
    print '\n'
    
    # CAPTCHA NETWORK #
    ###################
    # We will attept to train a network to identify digits written out on a
    # 5x5 grid @@@@@
    #          @...@
    #          @...@
    #          @...@
    #          @@@@@
    # above is an example of 0 digit. Our input will be a list (25 els) to
    # represent the number. Our output will be a list (10 el long)
    print 'CAPTCHA NETWORK-----------------------'
    # We want to identify digits so we first define the targets, numbers
    # between 0 and 9 represented as vectors of the identity matrix
    targets = [[1 if i==j else 0 for i in range(10)] for j in range(10)]

    # Lets now build the network, at this point it is not clear why we would
    # choose 5 hidden layer neurons. The choice of 10 outputs makes sense
    # since we are trying to represent 10 numbers 0-9. Maybe it is just
    # empirical that 25 dim space can be represented in 5 dim hidden neurons
    # and then read out with a high success rate.
    random.seed(0)
    input_size = 25 # 5 x 5 grid
    num_hidden = 5 # five neurons in the hidden layer
    output_size = 10 # 10 possible outputs for each input

    # each hidden neuron will have one weight per input plus a bias weight
    hidden_layer = [[random.random() for _ in range(input_size + 1)] 
                    for _ in range(num_hidden)]

    # each output neuron has one weight per hidden neuron input plus a bias 
    # weight
    output_layer = [[random.random() for _ in range(num_hidden + 1)] 
                    for _ in range(output_size)]

    network = [hidden_layer, output_layer]
    
    # Function to convert the raw digits to 1's and 0's
    def make_digit(raw_digit):
        return [1 if c == '1' else 0
                for row in raw_digit.split("\n")
                for c in row.strip()]
    
    inputs = map(make_digit, raw_digits)

    # Train the network using backpropagation #10000 iterations
    for _ in tqdm(range(10000)):
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)

    # We can look to see how well it predicts an input that was present in
    # the training set
    def predict(input):
        return feed_forward(network, input)[-1]

        

    print [round(prediction,3) for prediction in predict(inputs[7])]

    # next lets plot the input weights of the hidden layer. It is a 5x5 grid
    # of inputs onto the 5 hidden layer neurons
    # create a figure and set of axes 
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(12,5))
    # get the range over which the weights vary
    min_weight = min(min(ls) for ls in network[0])
    max_weight = max(max(ls) for ls in network[0])

    # set the min, max value to plot over as the min of the abs(wieghts).
    # This will round very negative or positive weights
    value = min(abs(min_weight),max_weight)
    
    for neuron in range(5):
        # get the neurons input weights
        weights = network[0][neuron]
        # convert weights to grid
        grid = [weights[row:(row+5)] for row in range(0,25,5)]
        # draw with seismic map
        image = axes[neuron].imshow(grid, interpolation='none',
                                    vmin=-value, vmax=value,
                                    cmap=cm.seismic)
        # title with neuron number
        axes[neuron].set_title('network[0][%d]' %(neuron))
        # add the bias as the x label
        axes[neuron].set_xlabel('Bias = %.2f' %(network[0][neuron][-1]))
    # adjust the subplot positions to allow for a color bar
    fig.subplots_adjust(right=0.75)
    # add color bar (x, y, width, height
    cbar_ax = fig.add_axes([0.8, 0.25, .05, 0.5]) 
    fig.colorbar(image, cax=cbar_ax) 
    plt.show()

