import numpy as np

class Lr():
	"""logistic regress"""
	def __init__(self, niter=1000, alpha=-1 ):
		self.niter = niter
		self.alpha = alpha
		self.beta = np.zeros((1,3))

	def train(self,X,y):
		X=np.hstack((X,np.ones((X.shape[0],1))))
		self.beta = np.zeros((X.shape[1],1))
		print('init cost: ', self.costfunction(X,y,self.beta))
		costs =  np.ones((self.niter,1))
		(grad,hessian) = self.gradient(X,y,self.beta)
		print('init grad: %s\n' %grad)
		print('inti hessian: %s\n' %hessian)
		for i in range(0,self.niter) :
			if (self.alpha<=0):
				self.beta = self.beta - (np.linalg.pinv(hessian)).dot(grad)
			else:
				self.beta = self.beta - 0.3*(grad)
			(grad,hessian) = self.gradient(X,y,self.beta)
			cost_t=self.costfunction(X,y,self.beta)
			costs[i]=cost_t
		print('final cost: ' ,self.costfunction(X,y,self.beta))
		print('final beta: ' ,self.beta)
		return (self.beta,costs)

	def test(self,example):
		N = example.shape[0]
		example=np.hstack((example,np.ones((example.shape[0],1))))
		return (example.dot(self.beta)>0.5)

	def volidation(self,X,y):
		N = X.shape[0]
		res = self.test(X)
		t_res = np.sum(res==y)
		print ("rate = %s / %s = %s \n" %(t_res, N, t_res/N))
		return (t_res/N)

	def costfunction(self,X,y,theta):
		it1= -((theta.T).dot(X.T)).dot(y)
		it2= np.sum(np.log(1+np.exp(X.dot(theta))))
		return it1+it2

	def gradient(self,X,y,theta):
		sigmoid = 1/(1+np.exp(-X.dot(theta)))
		#print('sigmoid shape: ', sigmoid.shape)
		grad = X.T.dot(sigmoid-y)/y.size
		first_order = (-np.sum(X * (y - sigmoid), 0, keepdims=True)).T
		p = np.diag((sigmoid * (1-sigmoid)).reshape(y.size))
		second_order = X.T .dot(p).dot(X)

		hessian = np.zeros((theta.size,theta.size))/y.size
		# for i in range(0,y.size) :
		# 	x=X[i,:]
		# 	hessian += (x.T).dot(x)*sigmoid[i,0]*(1-sigmoid[i,0])
		hessian = (X.T).dot(sigmoid*(1-sigmoid)*X)
		#print('hessian shape:', hessian.shape)
		return (grad,hessian)
		#return (first_order, second_order)
