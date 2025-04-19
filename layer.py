import numpy as np

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2,3,0.5]
'''
# Explanation of the dot product with an example:
# The dot product between the weights matrix and the inputs vector is calculated as follows:
# For each neuron (row in weights), multiply each weight by the corresponding input and sum them up.
# Example for the first neuron:
# (0.2 * 1) + (0.8 * 2) + (-0.5 * 3) + (1.0 * 2.5) = 0.2 + 1.6 - 1.5 + 2.5 = 2.8
# Add the corresponding bias to this sum: 2.8 + 2 = 4.8
# This process is repeated for all neurons.

# The weights matrix:
# [[ 0.2   0.8  -0.5   1.0  ],
#  [ 0.5  -0.91  0.26 -0.5  ],
#  [-0.26 -0.27  0.17  0.87 ]]

# The biases vector:
# [2, 3, 0.5]

# The result of np.dot(weights, inputs) + biases will be:
# [4.8, 1.21, 2.385]
'''

output = np.dot(weights, inputs)+biases

print(f"layer output:- {output}")
