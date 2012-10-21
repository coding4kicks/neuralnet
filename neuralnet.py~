#!/usr/bin/env python
# encoding: utf-8

# An Artificial Neural Network
# Scott Young - CSC 578 - Oct. 2012

import os
import sys
import math
import random
from optparse import OptionParser

class NeuralNet(object):
    """ An artificial neural network """

    def __init__(self, eta, error_margin, hidden_nodes, max_epochs, weight, input_file):
        """ Constructor for the neural network. """
        self.eta = eta                      # learning rate
        self.error_margin = error_margin    # error margin to correctly classify
        self.hidden_nodes = hidden_nodes    # number of hidden nodes
        self.max_epochs = max_epochs        # max number of epochs to train for
        self.max_weight = weight            # max random weight value
        self.min_weight = -weight           # min random weight value
        self.output_weights = []            # the output weights 

        # read in the input file
        self.training_data = []
        if not os.path.exists(input_file):
            print "Input file doesn't exist."
            sys.exit(2)
        with open(input_file, 'r') as file:
            for line in file:
                training_input = line.split(',')
                i = 0
                for input in training_input:
                    training_input[i] = float(input.rstrip())
                    i = i + 1
                self.training_data.append(training_input)

        # A list of a list of hidden unit's weights
        self.hidden_units =  [[] for i in range(len(self.training_data[0])-1)] 

        # Error if no training data
        if not self.training_data:
            print "No training data in file."
            sys.exit(2)

        # Set output weights
        i = 0
        while i < int(self.hidden_nodes):
            print self.hidden_nodes
            print "i: " + str(i)
            weight = random.uniform(self.min_weight, self.max_weight)
            self.output_weights.append(weight)
            i = i + 1
        wH0 = -1            # Threshold for hidden nodes
        # Append the hidden threshold to the end of the output's weights
        self.output_weights.append(wH0)

        # Set all hidden nodes' weights
        i = 0
        num_inputs = len(self.training_data[0]) - 1
        while i < int(self.hidden_nodes):
            j = 0
            while j < num_inputs:
                weight = random.uniform(self.min_weight, self.max_weight)
                self.hidden_units[i].append(weight)
                j = j + 1
            i = i + 1

        wI0 = -1            # Initial threshold for input nodes
        h1 = [.2, .1]
        h2 = [.1, -0.1]
        self.hidden_units = [h1, h2]

        # Append the input threshold to the end of each hidden unit's weightings
        for h in self.hidden_units:
            h.append(wI0)

        # need to add wH0/wI0 to weight vectors, and 1 to input vectors


    def train(self):
        """ Train the neural network. """

        training = True     # Status of training(Done = False)

        epoch = 1           # Current Epoch
        output = 0          # output for the output node
        out_error = 0       # error term for the output
        num_correct = 0     # the number of training examples classified correctly
        max_rmse = 0        # max error
        ave_rmse = 0        # ave error

        # a list of each hidden node's output value - initialized to zero
        hidden_output = [0 for i in range(len(self.hidden_units)+1)]    

        # a list of each hidden node's error
        hidden_error = [0 for i in range(len(self.hidden_units)+1)] 

        data = {}           # dictionary for data to print to screen

        # train the network
        while training:

            # zero out the number classified correct, the max rmse, and the ave rmse
            num_correct = 0
            max_rmse = 0
            ave_rmse = 0
            total_error_sq = 0
            
            # for each training example
            for training_example in self.training_data:

                # determine target (last item in list)
                #print "training example: " # TEST
                #print training_example # TEST
                target = training_example[len(training_example)-1]

                # create new example with last item to 1 for threshold
                example = training_example[:]
                example[len(training_example)-1] = 1

                ### PROPAGATE THE INPUT FORWARD ###

                # for each hidden unit, calculate its output value
                #print training_example # TEST
                i = 0
                while i < len(self.hidden_units):
                    #print self.hidden_units[i]
                    #print example
                    hidden_output[i] = calculate_sigmoid(self.hidden_units[i], example)
                    i = i + 1

                # set last hidden output to 1 for threshold
                hidden_output[len(hidden_output)-1] = 1
                #print "Hidden Outputs: " # TEST
                #print hidden_output # TEST

                # for the output unit, calculate its output value
                output = calculate_sigmoid(self.output_weights, hidden_output)
                #print "Output: " + str(output) # TEST

                # don't update waits if correct (break from loop)
                #print "Error " + str(math.fabs(target-output)) # TEST
                #print out_error # TEST
                #print "Output: " + str(output)
                #print "Target: " + str(target)
                if math.fabs(target - output) < self.error_margin:
                    num_correct = num_correct + 1
                    #print "Correct"
                    #print
                    continue

                ### PROPATE THE ERROR BACKWARD ###

                # for the output unit, calculate its error
                out_error = output * (1 - output) * (target - output)
                
                # update max_rmse if necessary
                if math.fabs(out_error) > max_rmse:
                    max_rmse = math.fabs(out_error)

                # for each hidden unit, calculate its error
                i = 0
                for h_out in hidden_output:
                    hidden_error[i] = h_out * (1 - h_out) * self.output_weights[i] * out_error
            
                    # update max_rmse if necessary
                    if math.fabs(hidden_error[i]) > max_rmse:
                        max_rmse = math.fabs(hidden_error[i])
                    i = i + 1


                # udate the average error
                total_error_squared = math.pow(out_error, 2)
                for error in hidden_error:
                    total_error_squared = total_error_squared + math.pow(error, 2)
                
                #print "hidden error:"
                #print hidden_error
                #print "output weights:"
                #print self.output_weights

                ### UPDATE ALL WEIGHTS ###

                # update all weights for the output
                i = 0
                for weight in self.output_weights:
                    #print out_error
                    weight_delta = self.eta * out_error * hidden_output[i]
                    self.output_weights[i] = weight + weight_delta
                    i = i + 1
                #print "new output weights: "
                #print self.output_weights

                # update the weights for each hidden node
                i = 0
                for h_node in self.hidden_units:

                    node = h_node[:]
                    #print "old hidden weights: " # TEST
                    #print h_node
                    
                    # update the hidden node' weights
                    error = hidden_error[i]
                    j = 0
                    for weight in node:
                        #print "example " + str(i) + "-" + str(j) + ":" + str(example[j])
                        weight_delta = self.eta * error * example[j]
                        node[j] = weight + weight_delta
                        #print "weight delta: " + str(weight_delta)
                        #print "hidden_node: " + str(h_node[j])
                        j = j + 1
                    self.hidden_units[i] = node
                    i = i + 1
                    #print "new hidden weights: " # TEST
                    #print h_node

            ave_rmse = math.sqrt(total_error_squared)

                #print #TEST
            #print "num correct " + str(num_correct)
            #print len(self.training_data)
            #print (num_correct / len(self.training_data))
            percent_correct = (float(num_correct) / len(self.training_data)) * 100
            #print "per correct " + str(percent_correct)
            # output the epoch's data and increment epoch    
            data["epoch"] = epoch
            data["max_RMSE"] = max_rmse
            data["ave_RMSE"] = ave_rmse
            data["correct"] = percent_correct
            print_output(data)
            print 
           
            # quit training if max epoch or 100% classified correct
            if epoch == self.max_epochs or percent_correct == 100:
                training = False
            
            epoch = epoch + 1

        
        # TEST print weights
        print "Final Output Weights: "
        print self.output_weights
        print
        print "Final Hidden Weights: "
        for h in self.hidden_units:
            print h
        print "-------------------------------"
        print

# Helper Functions
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

    # Parse command line options and arguments.
    usage = "usage: %prog [options] <input_file>"
    parser = OptionParser(usage)
    parser.add_option("-l", "-L", "--Eta", action="store", default=0.3,
                      dest="eta", help="Set the learning rate (Eta). [default: %default]")
    parser.add_option("-e", "-E", "--error", action="store", default=.05,
                      dest="error", help="Set maximum error margin. [default: %default]")
    parser.add_option("-p", "-P", "--epochs", action="store", default=10000,
                      dest="epochs", help="Set maximum number of epochs. [default: %default]")
    parser.add_option("-n", "-N", "--hiddenNodes", action="store",
                      dest="hiddenNodes", default=10,
                      help="Set the number of hidden nodes. [default: %default]")
    parser.add_option("-w", "-W", "--weight", action="store", default=0.05,
                      dest="weight", help="Set the max/min weights. [default: %default]")
    (options, args) = parser.parse_args()

    if len(args) == 0:
        parser.error("Must specify path to input file.")
    if len(args) > 1:
        parser.error("Can only specify 1 input file.")
    
    # Initialize network based upon parameters.
    ann = NeuralNet(options.eta, options.error, options.hiddenNodes, 
                    options.epochs, options.weight, args[0])
    ann.train()


if __name__ == "__main__":
    """ enable command line execution """
    sys.exit(main())
