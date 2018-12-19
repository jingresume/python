# -*- coding: utf-8 -*-

' read dataset module '

__author__ = 'Jing'

import numpy as np 
import pandas as pd 

def dataset_30_a():
	data = pd.read_csv("C:\\Users\\3142_2\\Desktop\\dataset\\xg3.0a.csv")
	X = data.loc[:,['density','suger']].as_matrix()
	y = data.loc[:,'quality'].as_matrix()[:,np.newaxis]
	return (X,y)

def dataset_fj():
	data = pd.read_csv("C:\\Users\\3142_2\\Desktop\\dataset\\fj.csv")
	X = data.loc[:,['density','suger']].as_matrix()
	y = data.loc[:,'quality'].as_matrix()[:,np.newaxis]
	return (X,y)

def dataset_iris():
	data = pd.read_csv("C:\\Users\\3142_2\\Desktop\\dataset\\lris.csv")
	X = data.loc[:,['X1','X2','X3','X4']].as_matrix()
	y = data.loc[:,'Y'].as_matrix()[:,np.newaxis]
	return (X,y)

def dataset_30():
	data = pd.read_csv("C:\\Users\\3142_2\\Desktop\\dataset\\xg3.0.csv", encoding='GB2312')
	#print (data)
	X = data.loc[:,['色泽','根蒂','敲声','纹理','脐部','触感','密度','含糖率']].as_matrix()
	X = np.vstack((['色泽','根蒂','敲声','纹理','脐部','触感','密度','含糖率'],X))
	y = data.loc[:,'好坏'].as_matrix()[:,np.newaxis]
	return (X,y)

def dataset_30_c():
	data = pd.read_csv("C:\\Users\\3142_2\\Desktop\\dataset\\xg3.0c.csv", encoding='GB2312')
	X = data.loc[:,['色泽','根蒂','敲声','纹理','脐部','触感','密度','含糖率']].as_matrix()
	y = data.loc[:,'好坏'].as_matrix()[:,np.newaxis]
	return (X,y)

def dataset_40():
	data = pd.read_csv("C:\\Users\\3142_2\\Desktop\\dataset\\xg4.0.csv")
	X = data.loc[:,['density','suger']].as_matrix()
	return X

def test(X,y):
	print('Train set X: ', X)
	print('Train set y: ', y)