import numpy as np
import debug_function as df 
from lr import Lr 
import dataset as ds 

def cross_validation(k,X,y,cls):
	tmp = y[0]
	c1 = np.where(y==tmp)[0]
	c2 = np.where(y!=tmp)[0]
	#X1 = X[c1,:]
	#y1 = y[c1,:]
	#X2 = X[c2,:]
	#y2 = X[c2,:]
	N = y.size
	NC = int(N/k)
	NC1 = int(c1.size/k)
	NC2 = int(c2.size/k)
	#Xi=np.zeros((k,NC1+NC2,X.shape[1]))
	#yi=np.zeros((k,NC1+NC2,y.shape[1]))
	#yi=np.empty((k,NC1+NC2,y.shape[1]),dtype = object)
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





def main():
	(X,y) = ds.dataset_iris()
	print("X shape :", X.shape)
	print("y shape :", y.shape)
	c1=np.where(y=='Iris-setosa')[0]
	c2=np.where(y=='Iris-versicolor')[0]
	c3=np.where(y=='Iris-virginica')[0]
	X1=X[np.hstack((c1,c2)),:]
	y1=np.vstack((np.ones((c1.size,1)),np.zeros((c2.size,1))))
	X2=X[np.hstack((c2,c3)),:]
	y2=np.vstack((np.ones((c2.size,1)),np.zeros((c3.size,1))))
	print("X1 shape :", X1.shape)
	print("y1 shape :", y1.shape)
	print("X2 shape :", X2.shape)
	print("y2 shape :", y2.shape)
	print("y dtype: ", y.dtype)
	cls = Lr(2000, 0.01)
	#cls = Lr()
	#cross_validation(10,X1,y1,cls)
	ho_rate = hold_out(np.arange(10),X2,y2,cls)
	cv_rate = cross_validation(10,X2,y2,cls)

	print('hold_out rate : ' , ho_rate)
	print('cv_rate : ', cv_rate)

	print('')




if __name__ == '__main__' :
	main()

