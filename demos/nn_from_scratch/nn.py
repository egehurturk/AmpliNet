import numpy as np
import random

class MLP:

    def __init__(self, n_inputs=3, hlayers=[3, 5], n_outputs=2):

        self.n_inputs = n_inputs
        self.hlayers = hlayers
        self.n_outputs = n_outputs

        self.layers = [self.n_inputs] + self.hlayers + [self.n_outputs]

        # iniate random weights
        self.weights = self._iniate_weights()
        self.activations = self._initiate_activations()
        self.derivatives = self._initiate_derivatives()
        
    
    def forward(self, inputs):
        '''
        a(1) = x
        h(i) = a(i-1) * w(i-1)
        a(i) = f(h(i))

        a(n) = f((a n-1)*w(i-1))
        '''
        a = inputs
        self.activations[0] = inputs # a[0] = x

        for i, w in enumerate(self.weights):
            # calculate net inputs (h)
            h = np.dot(a, w)
            # apply activation function
            a = self._sigmoid(h)
            self.activations[i+1] = a # save the activation
        return a

    def back_propagate(self, error):
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]

            # apply sigmoid derivative function
            delta = error * self._sigmoid_derivative(activations)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0],-1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)


    def gradient_descent(self, learningRate=1):
        """Learns by descending the gradient
        Args:
            learningRate (float): How fast to learn.
        """
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate

    def train(self, inputs, targets, epochs, learning_rate):
        """Trains model running forward prop and backprop
        Args:
            inputs (ndarray): X
            targets (ndarray): Y
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
        """
        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0

            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # activate the network!
                output = self.forward(input)

                error = target - output

                self.back_propagate(error)

                # now perform gradient descent on the derivatives
                # (this will update the weights
                self.gradient_descent(learning_rate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)

            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(sum_errors / len(items), i+1))

        print("Training complete!")
        print("=====")


    def _iniate_weights(self):
        weights = []
        for i in range(len(self.layers) - 1):
            w = np.random.rand(self.layers[i], self.layers[i+1])
            weights.append(w)
        return weights

    def _initiate_activations(self):
        ac = []
        for i in range(len(self.layers)):
            a = np.zeros(self.layers[i])
            ac.append(a)
        return ac

    def _initiate_derivatives(self):
        derivatives = []
        for i in range(len(self.layers) - 1): # as no of weights = nlayers - 1
            d = np.zeros((self.layers[i], self.layers[i+1]))
            derivatives.append(d)
        return derivatives

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        """Sigmoid derivative function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """
        return x * (1.0 - x)

    def _mse(self, target, output):
        """Mean Squared Error loss function
        Args:
            target (ndarray): The ground trut
            output (ndarray): The predicted values
        Returns:
            (float): Output
        """
        return np.average((target - output) ** 2)


if __name__ == '__main__':

    # create a dataset to train a network for the sum operation
    items = np.array([[random.random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2, [5], 1)

    # train network
    mlp.train(items, targets, 50, 0.1)

    # create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    # get a prediction
    output = mlp.forward(input)

    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))