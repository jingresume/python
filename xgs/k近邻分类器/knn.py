import debug_function as df 
import dataset as ds 
import numpy as np 
import matplotlib.pyplot as plt


def knn(X,y,k):
	edist = np.zeros((len(y),1))
	y = 2*y-1
	print(y)
	for i in np.arange(0.22,0.78,0.01):
		for j in np.arange(0.02,0.48,0.01):
			for l in range(0,len(y)):
				edist[l] = np.power(np.linalg.norm(np.array([i,j]) - X[l,:]),2)
			index = np.argsort(edist,axis=0)
			pe = np.sign(np.sum(y[index[0:k],:]))
			if pe == 1 :
				plt.plot(i,j,'.y')
			else:
				plt.plot(i,j,'.g')

	for i in range(0,17) :
		if y[i] == 1:
			plt.plot(X[i,0],X[i,1],'ob')
		else:
			plt.plot(X[i,0],X[i,1],'xb')

	plt.show()



def main():
	(X,y) = ds.dataset_30_a()
	#df.plotData(X,y)
	knn(X,y,3)


if __name__ == '__main__' :
	main()
