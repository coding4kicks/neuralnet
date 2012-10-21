#!/usr/bin/env python
# encoding: utf-8

# An Artificial Neural Network
# Scott Young - CSC 578 - Oct. 2012

import sys
import math
from optparse import OptionParser

class NeuralNet(object):
    """ An artificial neural network """

    def __init__(self, eta, error_margin, hidden_nodes, max_epochs, weight):
        """ Constructor for the neural network. """

        self.eta = eta                      # learning rate
        self.error_margin = error_margin    # error margin to correctly classify
        self.hidden_nodes = hidden_nodes    # number of hidden nodes
        self.max_epochs = max_epochs        # max number of epochs to train for
        self.max_weight = weight            # max random weight value
        self.min_weight = -weight           # min random weight value

        # hardcode training examples for xor
        number_inputs = 2
        t1 = [0, 1, 0]
        t2 = [0, 1, 1]
        t3 = [1, 0, 1]
        t4 = [1, 1, 0]
        self.training_data = [t1, t2, t3, t4]

        # initialize network
        number_hidden = 2
        wH0 = -1            # threshold for hidden nodes
        self.output_weights = [0.1, 0.1]
        #append the hidden threshold to the end of the output's weights
        self.output_weights.append(wH0)

        wI0 = -1            # threshold for input nodes
        h1 = [0.1, 0.1]
        h2 = [0.1, 0.1]
        self.hidden_units = [h1, h2]

        # append the input threshold to the end of each hidden unit's weightings
        for h in self.hidden_units:
            h.append(wI0)

        # need to add wH0/wI0 to weight vectors, and 1 to input vectors


    def train(self):

        # a list for the hidden output values - initialized to zero
        hidden_output = [0 for i in range(len(self.hidden_units)+1)]  
       
        # the output for the output node
        output = 0       

        # dictionary for data to print to screen
        data = {}
        data["epoch"] = 1
        data["max_RMSE"] = .5
        data["ave_RMSE"] = .5
        data["correct"] = 80

        training = True
        epoch = 1

        # train the network
        while training:

            # set the number classified correct to 0
            correct = 0
            
            # for each training example
            for training_example in self.training_data:

                print training_example
                # determine target (last item in list)
                target = training_example[len(training_example)-1]

                # set last item to 1 for threshold
                training_example[len(training_example)-1] = 1

                ### PROPAGATE THE INPUT FORWARD ###

                print training_example # TEST
                # for each hidden unit, calculate its output value
                i = 0
                while i < len(self.hidden_units):
                    hidden_output[i] = calculate_sigmoid(self.hidden_units[i],
                                                         training_example)
                    i = i + 1

                # set last hidden output to 1 for threshold
                hidden_output[len(hidden_output)-1] = 1
                print hidden_output # TEST

                # for the output unit, calculate its output value
                output = calculate_sigmoid(self.output_weights, hidden_output)
                print output # TEST

                ### PROPATE THE ERROR BACKWARD ###

                # for the output unit, calculate its error
                out_error = output * (1 - output) * (target - output)
                print out_error # TEST

                # for each hidden unit, calculate its error

                ### UPDATE ALL WEIGHTS ###

                correct = 100
                print_output(data)
                epoch = epoch + 1
                if epoch == self.max_epochs or correct == 100:
                    training = False

# Helper Function
def calculate_sigmoid(weight_vector, input_vector):
    """ Perform vector dot product then calculate sigmoid """

    # calculate dot product
    i = 0
    y = 0
    while i < len(weight_vector):
        y = y + (weight_vector[i] * input_vector[i])
        i = i + 1
     
    # calculate sigmoid
    sigmoid = 1 / (1 + math.pow(math.e, (-y)))
    return sigmoid

def print_output(data):
    """ Output the information for each epoch to the screen. """

    print ("***** Epoch " + str(data["epoch"]) + " *****")
    print ("Maximum RMSE: " + str(data["max_RMSE"]))
    print ("Average RMSE: " + str(data["ave_RMSE"]))
    print ("Percent Correct: " + str(data["correct"]) + "%")

def main():
    """ Run program from the command line. """

    # hard code initial values for testing
    eta = 0.1
    error_margin = .05
    hidden_nodes = 2
    max_epochs = 100
    weight = 5;

    # Initialize network based upon parameters.
    ann = NeuralNet(eta, error_margin, hidden_nodes, max_epochs, weight)
    ann.train()


if __name__ == "__main__":
    """ enable command line execution """
    sys.exit(main())
