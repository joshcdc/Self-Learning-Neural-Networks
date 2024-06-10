# Self-Learning-Neural-Networks
My journey to explore neural networks
This journey starts with me following the book: Neural Networks from Scratch in Python, by Harrison Kinsley and Daniel Kukiela

Log 1 - 10/6/2024
-----------------
I have uploaded the basic exoskeleton of the neural networks. Using OOP, the first thing I had thought of was to make each neuron separate objects, but I soon found out that making them in layers not only require less objects but allow the mathematical operations to be carried out easily.

The input layer is made with just an array of inputs, and each hidden layer is an object, utilising its own weights and biases, randomly generated at first, to create new outputs. 
<<!! Biases are set to zero at the beginning but this may cause the neural network to appear dead>>

I learned to use NumPy to carry out the many matrix operations easily and conveniently. The math behind Neural Networks was not a problem to understand, as it was taught in my current year's syllabus. This including matrices, tranposition and dot product.

I also learnt about activation functions, the benefits of ReLU and Sigmoid, as well as the Softmax function for output neurons.
<<!! To prevent exponentially large values in the Softmax function, we subtract all inputs by the largest input, so exponentiated values range from 0 < x <= 1. >>

The current page reached is 106/658, but we have only written 18 lines of code.
