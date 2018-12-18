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