# BATCHES, LAYERS, AND OBJECTS

# inputs are features from single sample
# we will pass a batch of inputs

# BATCH SIZE INCREASES, THE LINE (line that fits through the samples) WIGGLES MUCH LESS THAN BEFORE
import numpy as np

 # X are the input values, NOT the number of inputs
X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  # weight of each neuron, here: 4 inputs from X
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

layer1 = Layer_Dense(4,5)
layer2= Layer_Dense(5,2)

layer1_output = layer1.forward(X)
layer2_output = layer2.forward(layer1_output)

print(layer1_output)
print(layer2_output)




