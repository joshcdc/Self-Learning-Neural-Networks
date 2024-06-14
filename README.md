# Self-Learning-Neural-Networks
My journey to explore neural networks
This journey starts with me following the book: Neural Networks from Scratch in Python, by Harrison Kinsley and Daniel Kukiela

Log 1 - 10/6/2024
-----------------
I have uploaded the basic exoskeleton of the neural networks. Using OOP, the first thing I had thought of was to make each neuron separate objects, but I soon found out that making them in layers not only require less objects but allow the mathematical operations to be carried out easily.

The input layer is made with just an array of inputs, and each hidden layer is an object, utilising its own weights and biases, randomly generated at first, to create new outputs. 

I learned to use NumPy to carry out the many matrix operations easily and conveniently. The math behind Neural Networks was not a problem to understand, as it was taught in my current year's syllabus. This including matrices, tranposition and dot product.

I also learnt about activation functions, the benefits of ReLU and Sigmoid, as well as the Softmax function for output neurons. To prevent exponentially large values in the Softmax function, we subtract all inputs by the largest input, so exponentiated values range from 0 < x <= 1.

Log 2 - 12/6/2024
-----------------
Since the last upload, I have done 3 things:
I learned and applied loss calculation, using a method called Categorical Cross Entropy, which calculates the -log of the correct probability which inverses higher probability to lower loss values. To prevent errors, the predictions are clipped on both sides by a very small amount.

After loss calculation, I learned about accuracy, which calculates how many samples in a batch of samples were guessed accurately, as a decimal value.

Both loss and accuracy calculations took into account the format that the true values were given in, either categorical or one-hot encoded values.

Lastly, I learned how to do optimisation, which we used a vertical dataset from the nnfs library. Simple optimisation included tweaking the best case weights and biases slightly and comparing the new loss to best loss. This was significantly better than randomising weights and biases, but is not the best.

Log 3 - 14/6/2024
-----------------
This next part of the neural network was the most difficult so far. To reduce the required randomness of the neural network, I learned about backpropagation and partial derivatives. Unfortunately the explanation in the book was difficult for me to understand, so I spent the last evening watching multiple videos, including one by 3B1B, to understand the concept.

Though I fully understand the theory around the partial differentiaion used to backpropagate, it was quite difficult to follow along with the code shown in the book. I will come back to understand this again in the future when I am more confident.

The few edits we made were adding a backward method to the classes, allowing us to reverse the loss function to its derivatives.


