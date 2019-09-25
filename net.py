import numpy as np

#Define data sets
feature_set = np.array([[0,1,0], [0,0,1], [1,0,0], [1,1,0], [1,1,1]])
labels = np.array([[1,0,0,1,1]])
labels = labels.reshape(5,1)


#Keep random generator unchanged
np.random.seed(42)

#Define neural network parameters
weights = np.random.rand(3, 1)
bias = np.random.rand(1)
learning_rate = 0.05

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

for training_step in range(2000):
    #Feed Forward
    z = np.dot(feature_set, weights) + bias
    err = sigmoid(z) - labels
    if training_step % 100 == 0:
        print(err)

    #Backpropagation
    dCost_dY = err

    dY_dZ = sigmoid_derivative(z)

    dZ_dW = feature_set

    weights -= learning_rate * np.dot(dZ_dW.T, dCost_dY * dY_dZ)


#Prediction
test_set = np.array([0,0,1])
prediction_result = sigmoid(np.dot(test_set, weights) + bias)
print(prediction_result)