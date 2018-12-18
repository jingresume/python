import numpy as np 
import dataset as ds 
import debug_function as df 
import math

class Property:
	def __init__(self, idnum, name):
		self.is_continuity = False
		self.attribute = name
		self.subattributes=[]
		self.id = idnum

class TreeNode:
	def __init__(self, id):
		self.id = id
		self.attr = []
		self.child_id = []
		self.thres = []
		self.is_leaf = False

class DTree():
	def __init__(self, X, y):
		self.train_set = []
		self.attr_set = []
		self._data_process(X,y)
		self.Tree = []
		self.index = 0

	def train(self, method = 'Gini'):
		self._TreeGenerate(self.train_set, self.attr_set, method)

	def test(self, test_set):
		N = test_set.shape[0]
		res = []
		for i in range(0,N):
			example = test_set[i,:]
			root = self.Tree[0]
			while(root.is_leaf == False):
				index = 0
				#print (root.attr)
				is_continuity = False
				for a in self.attr_set:
					if root.attr == a.attribute:
						index = a.id
						is_continuity = a.is_continuity
				if is_continuity:
					if example[index]>root.thres :
						root = self.Tree[root.child_id[0]]
					else:
						root = self.Tree[root.child_id[1]]
				else:
					for i in range(0,len(root.thres)):
						#print(root.thres[i])
						if example[index] == root.thres[i]:
							root = self.Tree[root.child_id[i]]
							break

			res.append(root.attr)
		return res

	def validation(self, X, y):
		N = X.shape[0]
		res = self.test(X)
		t_res = np.sum(res==y.T)
		print ("rate = %s / %s = %s \n" %(t_res, N, t_res/N))
		return (t_res/N)

	def print_tree(self):
		index = 0;
		for node in self.Tree :
			print('index: ', index)
			index += 1
			print('id %s , attr %s, child_id %s, thres %s, is_leaf %s \n' %(node.id,node.attr,node.child_id,node.thres,node.is_leaf))

	def _TreeGenerate(self, train_set, attr_set, method):
		node = TreeNode(self.index)
		#第一种递归边界， train_set 样本属于同一类
		#s = set(train_set[:, train_set.shape[1]-1])
		dict = {}
		for i in range(0,train_set.shape[0]) :
			key = train_set[i,train_set.shape[1]-1]
			if dict.get(key) == None:
				dict[key] = 1
			else:
				dict[key] += 1 
		max_class = max(dict,key=dict.get)
		#print('max_class :', max_class)
		if (len(dict) == 1):	
			node.attr = max_class
			node.is_leaf = True
			self.Tree.append(node)
			#self.index+=1
			return
		#第二种递归边界， attr_set为空，或者train_set在attr_set中所有的取值相同
		if len(attr_set) == 0:
			node.attr = max_class
			node.is_leaf = True
			self.Tree.append(node)
			#self.index+=1
			return
		attr_index = [c.id for c in attr_set ]
		DA_same = True
		first_row = train_set[0,attr_index]
		#print ('first_row: ', first_row)
		for i in range(1,train_set.shape[0]):
			if (train_set[i,attr_index] == first_row).all():
				#print('train_set: ', train_set[i,attr_index] )
				continue
			else:
				DA_same = False
				break
		if DA_same :
			node.attr = max_class
			node.is_leaf = True
			self.Tree.append(node)
			#self.index+=1
			return
		#从A中选取最佳划分属性
		if method == 'Ent' :
			(a_star, thres_star) = self._bestAttr_Ent(train_set, attr_set)
		elif method == 'Gini' :
			(a_star, thres_star) = self._bestAttr_Gini(train_set, attr_set)
		node.attr = a_star.attribute
		node.thres = thres_star
		l = a_star.subattributes
		example = train_set[:,a_star.id] 
		Av = attr_set.copy()
		Av.remove(a_star)
		#节点的分支
		if a_star.is_continuity:
			p_index = np.where(example > thres_star) [0]
			n_index = np.where(example <= thres_star) [0]
			for i in range(0,2) :
				if i==0:
					list_index = p_index
				else:
					list_index = n_index 
				Dv = train_set[ list_index, : ]
				if len(Dv) == 0 :
			#第三种叶节点，某种属性值的样本数目为0， 则将该分支设置为叶节点
					self.index+=1
					leafnode =TreeNode(self.index)
					leafnode.is_leaf = True
					leafnode.attr = max_class
					self.Tree.append(leafnode)
					node.child_id.append(leafnode.id)
			#开始递归
				else:
					self.index+=1
					node.child_id.append(self.index)
					self._TreeGenerate(Dv,attr_set,method)
		else :
			for a in l :
				list_index = np.where(example == a) [0]
				Dv = train_set[ list_index, : ]
				if len(Dv) == 0 :
			#第三种叶节点，某种属性值的样本数目为0， 则将该分支设置为叶节点
					self.index+=1
					leafnode = TreeNode(self.index)
					leafnode.is_leaf = True
					leafnode.attr = max_class
					self.Tree.append(leafnode)
					node.child_id.append(leafnode.id)
			#开始递归
				else:
					self.index+=1
					node.child_id.append(self.index)
					#print (a_star.attribute) 
					self._TreeGenerate(Dv,Av,method)
				node.thres.append(a)
		self.Tree.append(node)
		self.Tree.sort(key = lambda x: x.id)



	def _data_process(self,X,y):
		property_set = X[0,:]
		data_one = X[1,:]
		for index in range(0,len(property_set)) :
			tmp_property = Property(index,property_set[index])
			s = set(X[1:,index])
			l = [c  for c in s]
			for a in l :
				list_index = []
				for i in range(0,X.shape[0]) :
					if X[i,index] == a :
						list_index.append(i)
				 
			if(isinstance(data_one[index],float)):
				tmp_property.is_continuity = True
				l.sort()
			tmp_property.subattributes = l
			self.attr_set.append(tmp_property)
		self.train_set = np.hstack((X[1:,:],y))


	def _bestAttr_Ent(self, train_set, attr_set):
		# attr_name  = [attr.attribute for attr in attr_set]
		# print('attr name: ', attr_name)
		# attr_continue  = [attr.is_continuity for attr in attr_set]
		# print('attr continue: ', attr_continue)
		N = train_set.shape[0]
		D_Ent = self._Ent(train_set)
		best_attr = attr_set[0]
		best_thres = None
		dict_attr = {}
		max_Gain = -10000
		for a in attr_set:
			example = train_set[:,a.id]
			s = set(example)
			l = [c  for c in s]
			Gain_D_a = D_Ent
			current_thres = None
			if a.is_continuity :
				l.sort()
				Ta = [(l[i]+l[i+1])/2 for i in range(0,len(l)-1)]
				#print ('Ta: ', Ta)
				max_Gain_c = -10000
				for thres in Ta:
					p_index = np.where(example > thres )[0]
					n_index = np.where(example <= thres )[0]
					Dp = train_set[p_index, :]
					Dn = train_set[n_index, :]
					Gain_c = D_Ent - (len(Dp)/N)*self._Ent(Dp) -(len(Dn)/N)*self._Ent(Dn) 
					if Gain_c > max_Gain_c :
						max_Gain_c = Gain_c
						current_thres = thres
				Gain_D_a = max_Gain_c
			else:
				for sub_attr in l :
					list_index = np.where(example == sub_attr)[0] 
					Dv = train_set[ list_index, : ]
					Gain_D_a -= (len(Dv)/N)*self._Ent(Dv)
			#print ('Gain_Da: ' ,Gain_D_a)
			if Gain_D_a > max_Gain :
				max_Gain = Gain_D_a
				best_attr = a
				if a.is_continuity :
					best_thres = current_thres
				else:
					best_thres = []
		return (best_attr,best_thres)

	def _bestAttr_Gini(self, train_set, attr_set):
		N = train_set.shape[0]
		D_Ent = self._Ent(train_set)
		best_attr = attr_set[0]
		best_thres = None
		dict_attr = {}
		min_Gini = 10000
		for a in attr_set:
			example = train_set[:,a.id]
			s = set(example)
			l = [c  for c in s]
			Gain_D_a = D_Ent
			current_thres = None
			if a.is_continuity :
				l.sort()
				Ta = [(l[i]+l[i+1])/2 for i in range(0,len(l)-1)]
				#print ('Ta: ', Ta)
				min_Gini_c = 10000
				for thres in Ta:
					p_index = np.where(example > thres )[0]
					n_index = np.where(example <= thres )[0]
					Dp = train_set[p_index, :]
					Dn = train_set[n_index, :]
					Gini_c = (len(Dp)/N)*self._Gini(Dp) + (len(Dn)/N)*self._Gini(Dn) 
					if Gini_c < min_Gini_c :
						min_Gini_c = Gini_c
						current_thres = thres
				Gain_D_a = min_Gini_c
			else:
				for sub_attr in l :
					list_index = np.where(example == sub_attr)[0] 
					Dv = train_set[ list_index, : ]
					Gain_D_a += (len(Dv)/N)*self._Gini(Dv)
			#print ('Gain_Da: ' ,Gain_D_a)
			if Gain_D_a < min_Gini :
				min_Gini = Gain_D_a
				best_attr = a
				if a.is_continuity :
					best_thres = current_thres
				else:
					best_thres = []
		return (best_attr,best_thres)

	def _indexOfattr(self, train_set, attr , index):
		list_index = []
		example = X[:,index]
		list_index = np.where(example = attr)
		if isinstance(attr,float):
			for i in range(0,X.shape[0]) :
				if X[i,index] == a :
					list_index.append(i)
		else:
			pass



	def _Ent(self, train_set):
		N = train_set.shape[0]
		M = train_set.shape[1]-1
		dict = {}
		for i in range(0,N):
			key = train_set[i,M]
			if dict.get(key) == None:
				dict[key] = 1
			else:
				dict[key] += 1 
		res = 0
		for value in dict.values():
			pk = value / N
			res-=pk*math.log(pk,2)
		return res

	def _Gini(self, train_set):
		N = train_set.shape[0]
		M = train_set.shape[1]-1
		dict = {}
		for i in range(0,N):
			key = train_set[i,M]
			if dict.get(key) == None:
				dict[key] = 1
			else:
				dict[key] += 1 
		res = 1
		for value in dict.values():
			pk = value / N
			res-=pk*pk
		return res




def main():
	(X,y) = ds.dataset_30()

	tree = DTree(X,y)
	tree.train(method = 'Ent' ) # method = Ent or Gini
	tree.print_tree()

	res = tree.test(X[1:,:])
	print('res: ', res)
	rate = tree.validation(X[1:,:],y)
	print('rate: ', rate)


if __name__ == '__main__' :
	main()

