# HIDDEN LAYER ACTIVATION FUNCTIONS

# STEP FUNCTION - THE OUTPUT WILL ALWAYS BE 1 OR 0. 
# 0 MEANS NEGATIVE OUTPUT WHILE 1 MEANS POSITIVE OUTPUT


# RELU ACTIVATION FUNCTION - RECTIFIED LINEAR ACTIVATION FUNCTION
# IT SAYS THAT IF X IS GREATER THAN 0 THE OUTPUT IS X, IF X IS LESS THAN 0 THEN THE OUTPUT IS 0

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)  # X = feature sets, y = classes/groups of sets

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(2,5)
activate1 = Activation_ReLU()

layer1.forward(X)
#print(layer1.output)
activate1.forward(layer1.output)
print(activate1.output)

