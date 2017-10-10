import numpy as np

def train_simple_neural_network():
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    Y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    train_neural_network(X,Y)

def compute_derivative(x):
    return x * (1-x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def train_neural_network(X,Y):

    np.random.seed()

    layer1_weights = 2*np.random.random((3,4)) - 1
    layer2_weights = 2*np.random.random((4,1)) - 1

    for i in range(0,500000):

        layer0 = X
        layer1 = sigmoid(np.dot(layer0 , layer1_weights))
        layer2 = sigmoid(np.dot(layer1 , layer2_weights))

        layer2_error = Y - layer2
        layer2_delta = layer2_error * compute_derivative(layer2_error)

        print str(np.mean(np.abs(layer2_error)))

        # layer1_error = np.dot(layer1_weights,layer2_delta)
        layer1_error = layer2_delta.dot(layer2_weights.T)
        layer1_delta = layer1_error * compute_derivative(layer1_error)

        layer2_weights =  layer1.T.dot(layer2_delta)
        layer1_weights = layer0.T.dot(layer1_delta)

if __name__ == "__main__":
    train_simple_neural_network()