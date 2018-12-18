import matplotlib.pyplot as plt
import numpy as np

def plotData(X,y):
	pi=np.where(y==1)
	ni=np.where(y==0)
	print(pi)
	print(ni)
	plt.figure(1)
	for i in pi :
		plt.plot(X[i,0],X[i,1],'bo')
	for i in ni :
		plt.plot(X[i,0],X[i,1],'r*')
	plt.show()

def plotDecisionBoundary(X,y,beta):
	pi=np.where(y==1)
	ni=np.where(y==0)
	print(pi)
	print(ni)
	plt.figure(1)
	for i in pi :
		plt.plot(X[i,0],X[i,1],'bo')
	for i in ni :
		plt.plot(X[i,0],X[i,1],'r*')
	t=[np.min(X[:,0]), np.max(X[:,0])]
	s=-(beta[0]*t+beta[2])/beta[1]
	#s=-(beta[0,0]+beta[1,0]*t)/beta[2,0]
	plt.plot(t,s)
	plt.show()

def plotCostFunction(costs):
	plt.figure(2)
	t=np.arange(costs.size)
	plt.plot(t,costs)
	plt.show()

def cross_validation(k,X,y,cls):
	tmp = y[0]
	c1 = np.where(y==tmp)[0]
	c2 = np.where(y!=tmp)[0]
	N = y.size
	NC = int(N/k)
	NC1 = int(c1.size/k)
	NC2 = int(c2.size/k)
	rate = 0;
	for i in range(0,k):
		index = np.hstack((c1[i*NC1:(i+1)*NC1],c2[i*NC2:(i+1)*NC2]))
		print (i, ': ', index) 
		rate+=hold_out(index,X,y,cls)
	return rate/k

def hold_out(index,X,y,cls):
	valid_set_x=X[index,:]
	valid_set_y=y[index,:]
	mask=[True]*y.shape[0]
	for i in index:
		mask[i] = False
	train_set_x=X[mask,:]
	train_set_y=y[mask,:]
	print('train x shape: ', train_set_x.shape)
	print('valid x shape: ', valid_set_x.shape)
	cls.train(train_set_x,train_set_y)
	rate =cls.validation(valid_set_x,valid_set_y)
	print('rate: ', rate)
	return rate