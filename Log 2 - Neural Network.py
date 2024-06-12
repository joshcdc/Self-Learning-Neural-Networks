import numpy as np
import matplotlib.pyplot as plt
import nnfs

from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data

nnfs.init()

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

class Loss:
    def calculate(self, output_data, y):
        sample_losses = self.forward(output_data, y)
        mean_loss = np.mean(sample_losses)
        return mean_loss
    
class CategoricalCrossEntropy(Loss):
    def forward(self, predicts, trues):
        samples = len(predicts)
        clip_predicts = np.clip(predicts, 1e-7, 1 - 1e-7)

        # categorical labels
        if len(trues.shape) == 1:
            confidences = clip_predicts[range(samples), trues]

        # one-hot encoded labels
        elif len(trues.shape) == 2:
            confidences = np.sum(clip_predicts * trues, axis=1)
        
        negative_log = -np.log(confidences)
        return negative_log

# trial data set
X, y = vertical_data(samples=100, classes=3)

# declaring layers, activation functions, and loss function.
dense1 = Layer(2,3)
activation1 = ReLU()
dense2 = Layer(3,3)
activation2 = Softmax()
loss_func = CategoricalCrossEntropy()

# initialising values
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

# optimisation
for i in range(10000):
    dense1.weights += 0.05 * np.random.randn(2,3)
    dense1.biases += 0.05 * np.random.randn(1,3)
    dense2.weights += 0.05 * np.random.randn(3,3)
    dense2.biases += 0.05 * np.random.randn(1,3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_func.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions==y)

    if loss < lowest_loss:
        print("New weights at iteration:", i, "loss:", loss, "accuracy:", accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()