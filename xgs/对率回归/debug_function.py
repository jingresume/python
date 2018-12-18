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