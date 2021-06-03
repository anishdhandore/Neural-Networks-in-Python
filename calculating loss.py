# CATEGORICAL CROSS ENTROPY
# taking negative log of the probability disribution

import math
import numpy

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] +
        math.log(softmax_output[1])*target_output[1] +
        math.log(softmax_output[2])*target_output[2])

print(loss)

'''
this is same as :

loss = -(math.log(softmax_output[0]))
print(loss)
'''

'''
an optimized way of doing it :

loss = 0
for i,j in zip(softmax_output, target_output):
    loss += -(numpy.log(i)) * j

print(loss)
'''