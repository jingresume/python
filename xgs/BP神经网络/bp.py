import numpy as np 
import dataset as ds
import debug_function as df 
import functools

class Bp():
	def __init__(self,method = 'std', alpha=1, hidden_layer_size = 10):
		self.method = method
		self.alpha = alpha
		self.v = None
		self.w = None
		self.hl = hidden_layer_size

	def train(self,X,y, iter = 5000):
		input_layer_size = X.shape[0]
		yl = [str(c) for c in y]
		num_labels = len(set(yl))
		print('num_labels: ', num_labels )
		self.v = np.random.random((X.shape[1]+1, self.hl))
		self.w = np.random.random((self.hl+1,num_labels))
		print('alpha: ', self.alpha)
		(J,v_grad,w_grad) = self.ComputeGradient(X,y,num_labels)
		print('init J: ', J)
		for i in range(0,iter):
			self.v = self.v + self.alpha*v_grad
			self.w = self.w + self.alpha*w_grad
			(J,v_grad,w_grad) = self.ComputeGradient(X,y,num_labels)
		print('final J: %s:' %J)

	def test(self,X):
		A1 = np.hstack((np.ones((X.shape[0],1)),X))
		Z2 = A1.dot(self.v)  # Z2 每一行对应一个example输出的隐层节点
		A2 = np.hstack((np.ones((Z2.shape[0],1)),df.sigmoid(Z2)))
		Z3 = A2.dot(self.w)  # Z3 每一行对应一个example输出的节点
		H = df.sigmoid(Z3)  # 每一行对应一个example的输出
		print('H: ', H)
		return H


	def ComputeGradient(self,X,y,num_labels):
		v_grad = np.zeros((self.v.shape))
		w_grad = np.zeros((self.w.shape))
		# forward
		A1 = np.hstack((np.ones((X.shape[0],1)),X))
		Z2 = A1.dot(self.v)  # Z2 每一行对应一个example输出的隐层节点
		A2 = np.hstack((np.ones((Z2.shape[0],1)),df.sigmoid(Z2)))
		Z3 = A2.dot(self.w)  # Z3 每一行对应一个example输出的节点
		H = df.sigmoid(Z3)  # 每一行对应一个example的输出

		m = X.shape[0]
		J = 0
		for i in range(0,m) :
			ty = np.zeros((num_labels,1))
			ty[y[i]] = 1
			a1 = A1[i,:][:,np.newaxis]
			a2 = A2[i,:][:,np.newaxis] 
			a3 = H [i,:][:,np.newaxis] 
			b2 = A2[i,:][:,np.newaxis]
			gi = (ty-a3)*df.sigmoidGradient(a3)
			eh = (self.w[1:,:].dot(gi) * df.sigmoidGradient(b2[1:,:]))
			w_grad = w_grad + b2.dot(gi.T)
			v_grad = v_grad + a1.dot(eh.T)
			J += np.sum((a3-ty)**2)/2

		J = J/m
		v_grad = v_grad/m
		w_grad = w_grad/m
		return (J, v_grad, w_grad)

	def costFunction(self, nn_params, input_layer_size, hidden_layer_size, num_labels, X, y):
		Theta1 = nn_params[0:hidden_layer_size*(input_layer_size + 1)].reshape((input_layer_size + 1, hidden_layer_size))
		Theta2 = nn_params[hidden_layer_size*(input_layer_size + 1) :].reshape((hidden_layer_size +1, num_labels))

		v_grad = np.zeros((Theta1.shape))
		w_grad = np.zeros((Theta2.shape))

		A1 = np.hstack((np.ones((X.shape[0],1)),X))  # -1
		Z2 = A1.dot(Theta1)  # Z2 每一行对应一个example输出的隐层节点
		A2 = np.hstack((np.ones((Z2.shape[0],1)),df.sigmoid(Z2))) # -1
		Z3 = A2.dot(Theta2)  # Z3 每一行对应一个example输出的节点
		H = df.sigmoid(Z3)  # 每一行对应一个example的输出
		m = X.shape[0]
		J = 0
		for i in range(0,m) :
			ty = np.zeros((num_labels,1))
			ty[y[i]] = 1
			a1 = A1[i,:][:,np.newaxis]
			a2 = A2[i,:][:,np.newaxis] 
			a3 = H [i,:][:,np.newaxis] 
			b2 = A2[i,:][:,np.newaxis]
			gi = (ty-a3)*df.sigmoidGradient(a3)
			eh = (Theta2[1:,:].dot(gi) * df.sigmoidGradient(b2[1:,:]))
			w_grad = w_grad + b2.dot(gi.T)
			v_grad = v_grad + a1.dot(eh.T)
			J += np.sum((a3-ty)**2)/2

		J = J/m
		v_grad = v_grad/m
		w_grad = w_grad/m
		grad = np.vstack((v_grad.reshape((v_grad.size,1)),w_grad.reshape((w_grad.size,1))))
		return (J,grad)

	def checkGradients(self):
		input_layer_size = 3
		hidden_layer_size = 5
		num_labels = 3
		m = 5

		Theta1 = df.debugInitializeWeights(input_layer_size, hidden_layer_size)
		Theta2 = df.debugInitializeWeights(hidden_layer_size, num_labels)

		X  = df.debugInitializeWeights(m-1, input_layer_size);
		y  = np.mod(range(0,m), num_labels).T;

		theta1 = Theta1.reshape((Theta1.size,1))
		theta2 = Theta2.reshape((Theta2.size,1))
		nn_params = np.vstack((theta1,theta2));

		costFunc = functools.partial(self.costFunction, input_layer_size=input_layer_size, hidden_layer_size=hidden_layer_size, num_labels=num_labels, X=X, y=y)

		(cost, grad) = costFunc(nn_params);
		numgrad = df.computeNumericalGradient(costFunc, nn_params)
		grad = -grad
		disp = np.hstack((grad,numgrad))
		print('grad and numgrad: ', disp)
		diff = np.linalg.norm(numgrad-grad) / np.linalg.norm(numgrad+grad)
		print('relative Difference: ', diff)




def main():
	(X,y) = ds.dataset_30_c() 
	print('X: ', X)
	print('y: ', y)
	cls = Bp(method = 'std', alpha=1, hidden_layer_size = 10)
	cls.checkGradients()
	cls.train(X,y,iter=10000)
	cls.test(X)



if __name__ == '__main__' :
	main()