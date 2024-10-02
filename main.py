import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # initialize sizes and learning rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # use seed function for reproducibility
        np.random.seed(1) # for some reason 1 is always more accurate than 0, 2, 3, etc.??
        
        # initialize weights
        self.weights_input_hidden = 2 * np.random.random((self.input_size, self.hidden_size)) - 1
        self.weights_hidden_output = 2 * np.random.random((self.hidden_size, self.output_size)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_of_sigmoid(self, x):
        return x * (1 - x)

    def forward(self, X):
        # forward propagation
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y):
        # calculate error
        error = y - self.final_output

        # back propagation
        final_output_delta = error * self.derivative_of_sigmoid(self.final_output)

        hidden_error = final_output_delta.dot(self.weights_hidden_output.T)
        hidden_output_delta = hidden_error * self.derivative_of_sigmoid(self.hidden_output)

        # update weights
        self.weights_hidden_output += self.hidden_output.T.dot(final_output_delta) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_output_delta) * self.learning_rate

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        return self.forward(X)

if __name__ == "__main__":
    # training data has 4 samples, 3 features each
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    # desired output 
    y = np.array([[0], [1], [1], [0]])

    # initialize and train neural network
    epochs = 10000
    nn = NeuralNetwork(input_size=3, hidden_size=4, output_size=1, learning_rate=0.1)
    nn.train(X, y, epochs=epochs)

    # output after training
    predictions = nn.predict(X)
    print(f"Output after {epochs} rounds of training:")
    print(predictions)