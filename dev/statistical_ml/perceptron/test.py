import random
import numpy as np
import perceptron
import perceptron_dual


def generate_dataset(rd_num = 0):
	arr = [[3,3,1], [4,3,1], [1,1,-1]]
	if rd_num > 0:
		for i in range(rd_num):
			x1 = random.randint(0, rd_num)
			x2 = random.randint(0, rd_num)
			y = (x1 * x2 > rd_num and 1) or -1
			arr.append([x1, x2, y])
	dataset = np.array(arr)
	X = dataset[:,:-1]
	Y = dataset[:,-1]
	return X, Y


X, Y = generate_dataset(2)
print(X)
print(Y)

# per = perceptron.perceptron()
# per.learning(X, Y)
# per.plt_learning_plane(X, Y)

per_dual = perceptron_dual.perceptron_dual()
per_dual.plt_learning_plane(X, Y, False)
per_dual.learning(X, Y)
per_dual.plt_learning_plane(X, Y)
