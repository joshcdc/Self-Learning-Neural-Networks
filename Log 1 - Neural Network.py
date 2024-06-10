import numpy as np

class Layer:
    def __init__(self, input_no, neuron_no):
        self.weights = 0.1*np.random.randn(input_no, neuron_no)
        self.biases = np.zeros((1, neuron_no))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    
class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Softmax:
    def forward(self, inputs):
        exponent = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exponent / np.sum(exponent, axis = 1, keepdims = True)
        self.output = probabilities