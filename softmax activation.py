# SOFTMAX ACTIVATION FUNCTION

# it helps us to identify the more relevant output

# first uses exponentiation to make negative output positive : e^x, x = layer output
# then normalizing values to not lose its meaning 

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

'''
layer_output = [12, -0.298, 2.385]

# first making sure that all values are positive
e = math.e
exponent_values = []

for i in layer_output:
    exponent_values.append(e**i)
print(exponent_values)

# normalizing the exponent values
base = sum(exponent_values)
normalized_values = []

for i in exponent_values:
    normalized_values.append(i/base)
print(normalized_values)
print(sum(normalized_values))
'''

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Softmax_Activation:
    def forward(self, inputs):
        # find the exponent value: e**input
        # axis = 0, columns are considered, axis = 1, rows are considered, keepdims = True, does not change the dimension of the array
        # inputs - np.max(inputs), so that exponent values aren't too big
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
        # next, normalizing the value 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

# layer 1 
layer1 = Layer_Dense(2,3) # no. of inputs = 2 since there are two axis. although you can have any number of neurons!
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)

# layer 2
layer2 = Layer_Dense(3,3) # (3,3) because of the dimensions of layer 1, i.e. (2,3)
activation2 = Softmax_Activation()

layer2.forward(activation1.output) # input is the output from the first layer
activation2.forward(layer2.output)

#print(len(activation2.output))
print(activation2.output[0:5]) # printing only the first 5 batches out of 300
