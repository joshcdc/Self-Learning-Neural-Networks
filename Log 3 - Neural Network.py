import numpy as np
import matplotlib.pyplot as plt
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()

class Layer:
    def __init__(self, input_no, neuron_no):
        self.weights = 0.1*np.random.randn(input_no, neuron_no)
        self.biases = np.zeros((1, neuron_no))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
        self.dinputs = np.dot(dvalues, self.weights.T)
    
class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exponent = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exponent / np.sum(exponent, axis = 1, keepdims = True)
        self.output = probabilities
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

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
    
    def backward(self, dvalues, trues):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(trues.shape) == 1:
            trues = np.eye(labels)[trues]
        
        self.dinputs = -trues/dvalues
        self.dinputs = self.dinputs/samples

class Softmax_CategoricalCrossEntropy:
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()
    def forward(self, inputs, trues):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, trues)
    def backward(self, dvalues, trues):
        samples = len(dvalues)

        if len(trues.shape) == 2:
            trues = np.argmax(trues, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), trues] -= 1
        self.dinputs = self.dinputs/samples


# trial data set
X, y = spiral_data(samples=100, classes=3)

# declaring layers, activation functions, and loss function.
dense1 = Layer(2,3)
activation1 = ReLU()
dense2 = Layer(3,3)
loss_activation = Softmax_CategoricalCrossEntropy()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)
print(loss_activation.output[:5])

print('loss:', loss)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)

print('acc:', accuracy)

loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)