import matplotlib.pyplot as plt
import numpy as np
import debug_function as df 
import dataset as ds 
from svm import *
from svmutil import *

def change_format(X):
	n = X.shape[0]
	m = X.shape[1]
	res = []
	for i in range(0,n):
		d = {}
		for j in range(0,m):
			d[j+1] = X[i,j]
		res.append(d)
	return res


def main():
	(X,y) = ds.dataset_30_a()
	x = change_format(X)
	prob = svm_problem(y,x)
	param = svm_parameter('-t 2 -c 100 -s 0  ')
	model = svm_train(prob,param)
	p_label, p_acc, p_val = svm_predict(y,x,model)
	print('p_acc: ', p_acc)
	print('p_label: ', p_label)
	print('p_val: ', p_val)

if __name__ == '__main__':
	main()