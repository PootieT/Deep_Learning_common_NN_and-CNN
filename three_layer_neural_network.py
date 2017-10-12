__author__ = 'tan_nguyen'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0: 
       return v
    return v/norm

def generate_data(data_set):
	'''
	generate data
	:return: X: input data, y: given labels
	'''
	np.random.seed(0)
	if data_set == "make_moons":
		X, y = datasets.make_moons(200, noise=0.20)
	elif data_set == "breast_cancer":
		data = load_breast_cancer()
		X = data["data"]
		y = data["target"]
	return X, y

def plot_decision_boundary(pred_func, X, y):
	'''
	plot the decision boundary
	:param pred_func: function used to predict the label
	:param X: input data
	:param y: given labels
	:return:
	'''
	# Set min and max values and give it some padding
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	h = 0.01
	# Generate a grid of points with distance h between them
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	# Predict the function value for the whole gid
	Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	# Plot the contour and training examples
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
	plt.show()


########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class NeuralNetwork(object):
	"""
	This class builds and trains a neural network
	"""
	def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
		'''
		:param nn_input_dim: input dimension
		:param nn_hidden_dim: the number of hidden units
		:param nn_output_dim: output dimension
		:param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
		:param reg_lambda: regularization coefficient
		:param seed: random seed
		'''
		self.nn_input_dim = nn_input_dim
		self.nn_hidden_dim = nn_hidden_dim
		self.nn_output_dim = nn_output_dim
		self.actFun_type = actFun_type
		self.reg_lambda = reg_lambda
		
		# initialize the weights and biases in the network
		np.random.seed(seed)
		self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
		self.b1 = np.zeros((1, self.nn_hidden_dim))
		self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
		self.b2 = np.zeros((1, self.nn_output_dim))

	def actFun(self, z, type):
		'''
		actFun computes the activation functions
		:param z: net input
		:param type: Tanh, Sigmoid, or ReLU
		:return: activations
		'''

		# YOU IMPLMENT YOUR actFun HERE
		# print "actFun function debug--the size of z input is: ", np.shape(z)

		type = type.lower()
		if type == 'tanh':
			activation = np.tanh(z)
		elif type == 'sigmoid':
			activation = 1.0/(1.0+np.exp(-z))
		elif type == 'relu':
			activation = np.maximum(0,z)
		else:
			print "Please choose one of three models: Tanh, Sigmoid, or ReLU"
			return None
		return activation

	def diff_actFun(self, z, type):
		'''
		diff_actFun computes the derivatives of the activation functions wrt the net input
		:param z: net input
		:param type: Tanh, Sigmoid, or ReLU
		:return: the derivatives of the activation functions wrt the net input
		'''

		# YOU IMPLEMENT YOUR diff_actFun HERE
		type = type.lower()
		if type == 'tanh':
			der_activation = 1.0 - (np.tanh(z))**2
		elif type == 'sigmoid':
			der_activation = 1.0/(1.0+np.exp(-z)) * (1.0 - 1.0/(1.0+np.exp(-z)))
		elif type == 'relu':
			der_activation = (z > 0)*1.0
		else:
			print "Please choose one of three models: Tanh, Sigmoid, or ReLU"
			return None
		return der_activation

	def feedforward(self, X, actFun):
		'''
		feedforward builds a 3-layer neural network and computes the two probabilities,
		one for class 0 and one for class 1
		:param X: input data
		:param actFun: activation function
		:return:
		'''

		# YOU IMPLEMENT YOUR feedforward HERE
		# print "feedforward funciton debug -- size of w1 is: ", np.shape(self.W1)
		# print "feedforward funciton debug -- size of X is: ", np.shape(X)
		# print "feedforward funciton debug -- size of b1 is: ", np.shape(self.b1)

		self.z1 = np.dot(X,self.W1) + np.tile(self.b1, (np.shape(X)[0],1))
		self.a1 = actFun(self.z1)
		self.z2 = np.dot(self.a1,self.W2) + np.tile(self.b2, (np.shape(self.a1)[0],1))
		exp_scores = np.exp(self.z2) 
		self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
		return None

	def calculate_loss(self, X, y):
		'''
		calculate_loss computes the loss for prediction
		:param X: input data
		:param y: given labels
		:return: the loss for prediction
		'''
		num_examples = len(X)
		self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
		# Calculating the loss

		# YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
		data_loss = -1.0 * sum(np.log(self.probs[np.arange(num_examples),y]))
		# Add regulatization term to loss (optional)
		data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
		return (1. / num_examples) * data_loss

	def predict(self, X):
		'''
		predict infers the label of a given data point X
		:param X: input data
		:return: label inferred
		'''
		self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
		return np.argmax(self.probs, axis=1)

	def backprop(self, X, y):
		'''
		backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
		:param X: input data
		:param y: given labels
		:return: dL/dW1, dL/b1, dL/dW2, dL/db2
		'''

		# IMPLEMENT YOUR BACKPROP HERE
		num_examples = len(X)
		delta3 = self.probs
		delta3[range(num_examples), y] -= 1
		dW2 = self.a1.T.dot(delta3)
		db2 = np.sum(delta3, axis = 0)
		delta2 = self.diff_actFun(self.z1, self.actFun_type) * delta3.dot(self.W2.T)
		dW1 = X.T.dot(delta2)
		db1 = np.sum(delta2, axis = 0)
		# dW2 = dL/dW2
		# db2 = dL/db2
		# dW1 = dL/dW1
		# db1 = dL/db1
		# print "backprop function debug--the size of diffact input is: ", np.shape(self.diff_actFun(self.z1, self.actFun_type))
		# print "backprop function debug--the size of delta3 input is: ", np.shape(delta3)
		# print "backprop function debug--the size of delta2 input is: ", np.shape(delta2)
		# print "backprop function debug--the size of b1 input is: ", np.shape(self.b1)
		# print "backprop function debug--the size of dw2 input is: ", np.shape(dW2)
		return dW1, dW2, db1, db2

	def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
		'''
		fit_model uses backpropagation to train the network
		:param X: input data
		:param y: given labels
		:param num_passes: the number of times that the algorithm runs through the whole dataset
		:param print_loss: print the loss or not
		:return:
		'''
		# Gradient descent.
		for i in range(0, num_passes):
			# Forward propagation
			self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
			# Backpropagation
			dW1, dW2, db1, db2 = self.backprop(X, y)

			# Add regularization terms (b1 and b2 don't have regularization terms)
			dW2 += self.reg_lambda * self.W2
			dW1 += self.reg_lambda * self.W1

			# Gradient descent parameter update
			self.W1 += -epsilon * dW1
			self.b1 += -epsilon * db1
			self.W2 += -epsilon * dW2
			self.b2 += -epsilon * db2

			# Optionally print the loss.
			# This is expensive because it uses the whole dataset, so we don't want to do it too often.
			if print_loss and i % 1000 == 0:
				print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

	def visualize_decision_boundary(self, X, y):
		'''
		visualize_decision_boundary plots the decision boundary created by the trained network
		:param X: input data
		:param y: given labels
		:return:
		'''
		plot_decision_boundary(lambda x: self.predict(x), X, y)


class DeepNeuralNetwork(NeuralNetwork):
	"""
	This class builds and trains a neural network
	"""
	def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim,  actFun_type='tanh', reg_lambda=0.01, seed=0):
		'''
		:param nn_input_dim: input dimension
		:param nn_hidden_dim: a list of numbers representing hidden layer units
		:param nn_output_dim: output dimension
		:param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
		:param reg_lambda: regularization coefficient
		:param seed: random seed
		'''
		self.nn_input_dim = nn_input_dim
		self.nn_hidden_dim = nn_hidden_dim
		self.nn_output_dim = nn_output_dim
		self.actFun_type = actFun_type
		self.reg_lambda = reg_lambda
		
		# initialize the weights and biases in the network
		self.layers = []
		self.layers = [Layer(nn_input_dim, nn_hidden_dim[0],seed)]
		if len(nn_hidden_dim) > 1:
			for i in range(len(nn_hidden_dim)-1):
				self.layers.append(Layer(nn_hidden_dim[i], nn_hidden_dim[i+1],seed))
		self.layers.append(Layer(nn_hidden_dim[-1], nn_output_dim,seed))
		
	def feedforward(self, X, actFun):
		'''
		feedforward builds a multi-layer neural network and computes one hot encoding 
		probabilities,
		:param X: input data
		:param actFun: activation function
		:return:
		'''
		for i in range(len(self.layers)):
			self.layers[i].feedforward(X, actFun)
			X = self.layers[i].a
		exp_scores = np.exp(self.layers[-1].z)
		self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
		return None

	def calculate_loss(self, X, y):
		'''
		calculate_loss computes the loss for prediction
		:param X: input data
		:param y: given labels
		:return: the loss for prediction
		'''
		num_examples = len(X)
		self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
		# Calculating the loss

		# YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
		data_loss = -1.0 * sum(np.log(self.probs[np.arange(num_examples),y]))

		# Add regulatization term to loss (optional)
		W_sum = 0
		for i in range(len(self.layers)):
			W_sum += np.sum(np.square(self.layers[i].W))
		data_loss += self.reg_lambda / 2 * W_sum
		return (1. / num_examples) * data_loss

	def backprop(self, X, y):
		'''
		backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
		:param X: input data
		:param y: given labels
		:return: dL/dW1, dL/b1, dL/dW2, dL/db2
		'''
		dW = range(len(self.layers))
		db = range(len(self.layers))
		num_examples = len(X)

		delta = self.probs
		delta[range(num_examples), y] -= 1
		dW[-1] = self.layers[-2].a.T.dot(delta)
		db[-1] = np.sum(delta, axis = 0)
		for i in reversed(range(len(self.layers)-1)):
			if i == 0:
				dW[i],db[i], delta = self.layers[i].backprop(X, delta, self.layers[i+1].W, self.diff_actFun, self.actFun_type)
			else:
				dW[i],db[i], delta = self.layers[i].backprop(self.layers[i-1].a, delta, self.layers[i+1].W, self.diff_actFun, self.actFun_type)
		return dW, db

	def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
		'''
		fit_model uses backpropagation to train the network
		:param X: input data
		:param y: given labels
		:param num_passes: the number of times that the algorithm runs through the whole dataset
		:param print_loss: print the loss or not
		:return:
		'''
		loss_vec = []
		iter_vec = []
		# Gradient descent.
		for i in range(0, num_passes):
			# Forward propagation
			self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
			# Backpropagation
			dW, db = self.backprop(X,y)

			# Add regularization terms (b1 and b2 don't have regularization terms)
			for j in range(len(self.layers)):
				dW[j] += self.reg_lambda * self.layers[j].W

			# Gradient descent parameter update
			for j in range(len(self.layers)):
				self.layers[j].W += -epsilon * dW[j]
				self.layers[j].b += -epsilon * db[j]
			# Optionally print the loss.
			# This is expensive because it uses the whole dataset, so we don't want to do it too often.
			
			if print_loss and i % 1000 == 0:
				print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))
				loss_vec.append(self.calculate_loss(X, y))
				iter_vec.append(i)
		return iter_vec, loss_vec
					

class Layer(object):
	"""
	This class builds layers for deep neuro-network
	"""
	def __init__(self, input_dim, output_dim, seed=0):
		'''
		:param input_dim: input dimention of the layer
		:param output_dim: output dimention of the layer
		'''
		# initialize the weights and biases in the network
		np.random.seed(seed)
		self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
		self.b = np.zeros((1, output_dim))

	def feedforward(self, X, actFun):
		'''
		feedforward builds 1-layer neural network and computes the two probabilities,
		one for class 0 and one for class 1
		:param X: input data
		:param actFun: activation function
		:return:
		'''
		self.z = np.dot(X,self.W) + np.tile(self.b, (np.shape(X)[0],1))
		self.a = actFun(self.z)
		return None

	def backprop(self, a_after, delta_prev, W_prev, diff_actFun, actFun_type):
		'''
		backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
		:param a_after: a from layer l-1
		:param delta_prev: delta from layer l+1
		:param diff_actFun: differential of activation function
		:param actFun_type: the type of activation function used
		:return: dL/dW1, dL/b1, dL/dW2, dL/db2
		'''
		# print "size of z: ", np.shape(self.z)
		# print "size of a: ", np.shape(self.a)
		# print "size of w: ", np.shape(self.W)
		# print "size of b: ", np.shape(self.b)
		# print "size of delta_prev: ", np.shape(delta_prev) 
		# print "size of a_after: ", np.shape(a_after)
		delta = diff_actFun(self.z, actFun_type) * delta_prev.dot(W_prev.T)
		dW = a_after.T.dot(delta)
		db = np.sum(delta, axis = 0)
		return dW, db, delta

def main():
	#++=======================================================================
	# # generate and visualize Make-Moons dataset
	# X, y = generate_data("make_moons")
	# plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
	# plt.show()

	# old model
	# model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3 , nn_output_dim=2, actFun_type='tanh')
	# new model with flexible input
	# model = DeepNeuralNetwork(nn_input_dim=2, nn_hidden_dim=[6,6,6,6,6,6,6,6,6,6,6,6,6,6,6], nn_output_dim=2, reg_lambda=0.001, actFun_type='tanh')
	# model.fit_model(X,y,0.0001)
	# model.visualize_decision_boundary(X,y)

	#=========================================================================
	# generate and visualize breast cancer dataset
	X,y = generate_data("breast_cancer")
	# print np.shape(X[1:30,:])
	X = normalize(X)
	model = DeepNeuralNetwork(nn_input_dim=30, nn_hidden_dim=[10,5], nn_output_dim=2, reg_lambda=0.0001, actFun_type='tanh')
	iter_vec, loss_vec = model.fit_model(X,y,0.0001)
	plt.scatter(iter_vec, loss_vec)
	plt.plot(iter_vec, loss_vec)
	plt.title('training loss over iterations')
	plt.xlabel('Ierations')
	plt.ylabel('Loss')
	plt.show()
if __name__ == "__main__":
	main()