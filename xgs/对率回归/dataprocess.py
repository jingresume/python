import pandas as pd
import numpy as np
import dataset as ds 
import debug_function as df
import matplotlib.pyplot as plt
from numpy import linalg
from lr import Lr

		

def main():
	cls = Lr(10000);
	# (X,y)=ds.dataset_30_a()
	(X,y)=ds.dataset_fj()
	print('X shape:  ' ,X.shape)
	print('Y shape:' , y.shape)
	(beta,costs)=cls.train(X,y)
	print(cls.test(X))
	print(cls.volidation(X,y))
	df.plotDecisionBoundary(X,y,beta)
	df.plotCostFunction(costs)


if __name__ == '__main__':
	main()