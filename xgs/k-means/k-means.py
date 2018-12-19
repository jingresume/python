import debug_function as df 
import dataset as ds 
import numpy as np 
import matplotlib.pyplot as plt
import random

def kmeans(X,k):
	index = [random.randint(0,9) for i in range(0,k) ]
	u = X[index,:]
	m = X.shape[0]
	niter = 0
	while True :
		C = [[] for c in range(0,k)]
		for j in range(0,m):
			min = 10000
			min_index = -1
			for i in range(0,k):
				d = np.linalg.norm(X[j,:]-u[i,:])
				if d < min:
					min = d
					index = i
			C[index].append(X[j,:])
		converge_flag = True
		for i in range(0,k):
			s = np.zeros((1,X.shape[1]))
			for j in range(0,len(C[i])):
				s += C[i][j]
			ui = s/len(C[i])
			if np.linalg.norm(ui - u[i]) > 0.000001:
				u[i] = ui
				converge_flag = False
		niter += 1
		if converge_flag:
			print('niter: ', niter)
			break
	return C





def main():
	X = ds.dataset_40()
	k=3
	C = kmeans(X,k)
	ch = ['ob', '*g', '+r', '.y', '>b']
	for i in range(0,k) :
		print (C[i])
		for ci in C[i]:
			plt.plot(ci[0], ci[1], ch[i])
	plt.show()


if __name__ == '__main__' :
	main()
