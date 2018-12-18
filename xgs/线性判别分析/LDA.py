import numpy as np 
import dataset as ds 
import debug_function as df 
import matplotlib.pyplot as plt

def LDA(X,y):
	c1 = np.where(y==1)[0]
	c2 = np.where(y==0)[0]
	X1 = X[c1,:]
	X2 = X[c2,:]
	u0 = (np.mean(X1, axis = 0))[:,np.newaxis]
	u1 = (np.mean(X2, axis = 0))[:,np.newaxis]
	delta1 = np.cov(X1, rowvar=False)
	delta2 = np.cov(X2, rowvar=False)
	print (u0.shape)
	print (u1.shape)
	#print (delta1)
	#print (delta2)
	Sw = delta1+delta2
	(U,s,VT) = np.linalg.svd(Sw)
	S = np.diag(s)
	Sw_inv = (VT.T).dot(np.linalg.inv(S)).dot(U.T)
	w = Sw_inv.dot(u0-u1)
	return w



def main():
	(X,y)=ds.dataset_30_a()
	w = LDA(X,y)
	print('w: ', w)
	df.plotLDA(X,y,w)


if __name__ == '__main__' :
	main()