import numpy as np
import perceptron
import perceptron_dual


def create_dataset():
	dataset = np.array([[3,3,1], [4,3,1], [1,1,-1]])
	X = dataset[:,:-1]
	Y = dataset[:,-1]
	return X, Y


X, Y = create_dataset()
print(X)
print(Y)

per = perceptron.perceptron()
per.classify(X, Y)

per_dual = perceptron_dual.perceptron_dual()
per_dual.classify(X, Y)
