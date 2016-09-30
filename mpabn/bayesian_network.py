"""bayesian_network.py

The purpose of this model is to assist with the construction of Bayesian Networks
designed for modeling the effects of a molecular mechanism. The goal is to learn
about this molecular mechanism by analysing the evidence that we have. We can then
perform a genome wide association compariang genotype with the combined evidence
of the molecular mechanism. The principal assumption behind this method is that
there is a molecular mechanism whose effects are observed across multiple phenotypes.

Here is an example network:
X1 --> X2 --> X3 --> X6
X2 --> X4 --> X3
X2 --> X5

X1 = Genotype
X2 = Molecular Mechanism
X3,X4,X5,X6 are pheontypes summarizing the disease

Goal: Learn X2 using the information in X3,X4,X5,X6 using Hard Expectation Maximiztion
Note: PCA will give similar results for Gaussian Networks without interactions

Features
--------
* Supports Linear Gaussian Nodes, Negative Binomial, Sigmoid, and categorical nodes
* Hard EM Algorithm for learning single latent (Currently it is limited to a Bernoulli Random Variable)

Example
-------



"""
from scipy.stats import bernoulli
from scipy.stats import uniform
from scipy.stats import norm
from scipy.stats import binom
import pandas as pd
import math
import random
import numpy as np
import scipy.stats as stats

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

from helpers import all_true

class BayesianNetwork:
	def __init__(self):
		pass
		
	def set_nodes(self,nodes):
		assert type(nodes) == list
		#validate that all are nodes and all names are distinct
		num_is_latent = 0
		names = {}
		for node in nodes:
			assert isinstance(node,_Node)
			if node.is_latent:
				num_is_latent += 1
			
			assert not names.has_key(node.name)
			names[node.name] = 1
			
		assert num_is_latent <= 1

		#compute topological sort
		self.__dfs__(nodes)
	
	def __dfs__(self,nodes):
		for node in nodes:
			node.tmp_mark = False
			node.perm_mark = False
			
		self.nodes = []
		for node in nodes:
			if not node.tmp_mark and not node.perm_mark:
				self.__visit__(node)
	
	def __visit__(self,node):
		if node.tmp_mark:
			raise Exception("Not a DAG")
			
		if not node.tmp_mark and not node.perm_mark:
			node.tmp_mark = True
			for child in node.children:
				self.__visit__(child)
			node.perm_mark = True
			node.tmp_mark = False
			self.nodes.insert(0,node)
	
	def print_network(self):
		for node in self.nodes:
			print str(node)
		
	#forward sample
	def forward_sample(self, n=1):
		sample = None
		for i in range(0,n):
			s = dict()
			for node in self.nodes:
				par_dict = dict()
				for parent in node.parents:
					par_dict[parent.name] = [ s[parent.name] ]
				s[node.name] = node.simulate(par_dict)
			
			if sample is None:
				sample = dict()
				for key,value in s.iteritems():
					sample[key] = list()
					sample[key].append(value)
			else:
				for key,value in s.iteritems():
					sample[key].append(value)	
		return pd.DataFrame(sample)
				
	#mle learning
	def mle(self,data):
		assert type(data) == pd.DataFrame
		for node in self.nodes:
			node.mle(data)
	
	def	joint_prob(self,dict_vals,log=True):
		assert self.nodes is not None
		log_joint = 0.0
		for node in self.nodes:
			print "joint " + str(node)
			log_joint += node.prob(dict_vals,log=True)
		
		if log:
			return log_joint
		else:
			return np.exp(log_joint)

#Note that the underscore makes this class private
class _Node:
	
	def __init__(self,name):
		self.type = "Node"
		self.name = name
		self.params = None
		self.children = []
		self.parents = []
		self.parent_names = {}
		self.children_names = {}
		self.limits = []
		#Current specification will only allow 1 node to be latent
		#Learning will be used ONLY children of this Node
		self.is_latent = False
	
	def set_params(self,params):
		assert params is not None
		self.params = params
			
	def is_root(self):
		return len(self.parents) == 0
		
	def is_leaf(self):
		return len(self.children) == 0
		
	def add_parent(self,parent):
		assert isinstance(parent,_Node)
		#add the parent to this node's parents
		#add this node to the parent's children
		self.parents.append(parent)
		self.parent_names[parent.name] = None
		parent.children.append(self)
		parent.children_names[self.name] = None
		
	def add_child(self,child):
		assert isinstance(child,_Node)
		#add the child to this node's children
		#add this node to the childs's parents
		self.children.append(child)
		self.children_names[child.name] = None
		child.parents.append(self)
		child.parent_names[self.name] = None
	
	def __str__(self):
		return "Name: {0}, Children: {1}".format(self.name, str(self.children_names.keys()))
		
	def mle(self):
		raise Exception("Not supported")
	
	def prob(self,log=True):
		raise Exception("Not supported")
			
class DiscreteNode(_Node):
	def __init__(self,name,values):
		assert name is not None
		assert type(values) is list
		assert len(values) > 1	
		_Node.__init__(self,name)
		# The number of values the discrete variable takes on
		self.num_vals = len(values)
		self.values = values
	
	def add_parent(self,parent):
		assert isinstance(parent,DiscreteNode)
		_Node.add_parent(self,parent)
	
	def set_params(self,params_df):
		"""params_df are a pandas dataframe
		must contain all the parents and itself in columns and an additional column for prob
		"""
		assert isinstance(params_df,pd.DataFrame)
		assert params_df.shape[1] == 2 + len(self.parents)
		assert self.name in params_df.columns
		assert "prob" in params_df.columns
		
		num_vals_in_df = self.num_vals
		for parent in self.parents:
			assert parent.name in params_df.columns
			num_vals_in_df *= parent.num_vals
		
		#make sure the number of rows are correct
		assert num_vals_in_df == params_df.shape[0]
		
		
		#make sure that when we sum across the values of self
		#that they sum to 1
		parent_names = self.parent_names.keys()
		if len(parent_names) > 0:
			for p in params_df.groupby(parent_names).sum()['prob']:
				np.testing.assert_almost_equal(p,1.0)
		else:
			np.testing.assert_almost_equal(params_df['prob'].sum(),1.0)		
		_Node.set_params(self,params_df)
		assert isinstance(self.params,pd.DataFrame)
	
	def prob(self,dict_vals,log=True):
		#make sure data has been input correctly
		assert self.params is not None
		assert dict_vals.has_key(self.name)
		for parent_name in self.parent_names.keys():
			assert dict_vals.has_key(parent_name)
		
		#find the matching probability
		prob = None
		for index,row in self.params.iterrows():
			keep = True
			value = dict_vals[self.name]
			if row[self.name] != value:
				keep = False
			
			if keep:	
				for variable in self.parent_names.keys():
					value = dict_vals[variable]
					if row[variable] != value:
						#print "Row" + str(row)
						#print variable
						#print value
						keep = False
						break
			
			if keep:
				prob = row["prob"]
				#we found a match so break
				break
		
		assert prob is not None
		
		if log:
			return np.log(prob)
		else:
			return prob	
	
	def mle(self,data):
		assert type(data) is pd.DataFrame
		parent_names = self.parent_names.keys()
		parent_names_and_self = [self.name]
		parent_names_and_self.extend(parent_names)
		#verify that the data fram contains the columns that we need
		assert len(data.columns.intersection(parent_names_and_self)) == len(parent_names) + 1
		
		
		#get columns of interest
		data_sub = data[parent_names_and_self]
		
		#TODO: check that the ranges match
		
		assert len(parent_names_and_self) > 0
		
		if len(parent_names_and_self) == 1:
			vals_count_dict = dict()
			list_of_vals_in_df = [x for x in data_sub[self.name]]
			#print list_of_vals_in_df
			#initialize dictionary
			for self_val in self.values:
				vals_count_dict[self_val] = 0.0
			
			#count observations
			total = 0.0
			for val in list_of_vals_in_df:
				assert vals_count_dict.has_key(val)
				vals_count_dict[val] += 1.0
				total += 1.0
			
			#create param_dict
			param_dict = dict()
			param_dict["prob"] = []
			param_dict[self.name] = []
			for key,val in vals_count_dict.iteritems():
				param_dict["prob"].append(val/total)
				param_dict[self.name].append(key)
			
			#create params dataframe
			params = pd.DataFrame(param_dict)
			self.set_params(params)
		
		else:
			
			#group by the parents
			df_parents_groups = data_sub.groupby(parent_names)
			params_res = dict()
			#initialize
			params_res["prob"] = list()
			for name in parent_names_and_self:
				params_res[name] = list()
	
			for group in df_parents_groups:
				parent_vals = group[0]
				assert len(parent_vals) == len(parent_names)
				g_value = group[1]
				counts = g_value.groupby([self.name]).count()[parent_names[0]]
				freqs = counts / counts.sum()
	
				#insert child and parent vals into dict results
				for self_val in self.values:
					self_prob = freqs.get(self_val)
					params_res["prob"].append(self_prob)
					params_res[self.name].append(self_val)
			
					#insert parent values	
					for index in range(0,len(parent_names)):
						par_name = parent_names[index]
						par_val = parent_vals[index]
						params_res[par_name].append(par_val)
		
	
			params = pd.DataFrame(params_res)
			self.set_params(params)
			
			
		
	
	def simulate(self,parents_vals=None):
		#print parents_vals
		if len(self.parent_names) > 0:	
			#make sure input is correct
			assert type(parents_vals) is dict
			for parent_name,val in parents_vals.iteritems():
				assert self.parent_names.has_key(parent_name)
				assert type(val) is list
				assert len(val) == 1
		else:
			parents_vals = dict()
			parents_vals[self.name] = self.values
		
		assert parents_vals is not None
		
		#subset data frame to only include parents (or only itself if is a root node)
		df_tmp_1 = self.params[parents_vals.keys()]
		#match rows to parent values
		df_tmp_2 = df_tmp_1.isin(parents_vals)
		#create a logical index for those rows
		ind = df_tmp_2.apply(all_true,axis=1)
		#subset the parameter data frame to find matching event
		matching_events = self.params.loc[ind]
		assert matching_events.shape[0] == self.num_vals, "Not enough events specified for {0}".format(str(self.params))
		#get the probabilites for those indices
		probs = matching_events['prob'].tolist()
		#random sample and get the index in the dataframe for that probability
		random_index = (np.random.multinomial(1,probs,size=1) == 1).tolist()[0]
		#get the value of this variable that that index corresponds to
		simulated_val = self.params.loc[ind][self.name].loc[random_index].iloc[0]
		return simulated_val
		
class GaussianNode(_Node):
	def __init__(self,name):
		_Node.__init__(self,name)
		self.type = "GaussianNode"
	
	def check_parents(self):
		for parent in self.parents:
			assert isinstance(parent, DiscreteNode) 
		
	def set_mean(self,mean):
		assert mean != None
		self.mean = mean
	
	def set_std(self,sigma):
		assert sigma != None
		self.sigma = sigma
		
	def simulate(self,n):
		sample = norm.rvs(loc=self.mean,scale=self.sigma,size=n)
		if(n == 1):
			return sample[1]
		else:
			return sample
		
	
class LinearGaussianNode(_Node):
	
	def __init__(self,name):
		_Node.__init__(self,name)
		self.type = "LinearGaussianNode"
	
	def set_variance(self,variance):
		assert len(var) == 0
		self.params.variance = variance
		
	def set_betas(self,betas):
		assert len(betas) > 0
		self.params.betas = betas

	def params_are_set(self):
		betas = self.params.betas
		variance = self.params.variance
		if betas == None or variance == None:
			return False
		elif len(betas) > 0 and len(variance) == 0:
			return True
		else:
			return False
			
	def simulate(self,x=None,n=1):
		assert x != None
		assert len(x) + 1 == len(self.params.betas)
		
