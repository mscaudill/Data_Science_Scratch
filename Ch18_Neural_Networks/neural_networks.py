import math
from DS_Scratch.Ch4_Linear_Algebra import dot_product

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
                     dot(output_deltas, [n[i] for n in output_layer])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # update the weights (j) of the hidden layer neurons (i)
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector) + [1]:
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
    print '--------------------------------------'



