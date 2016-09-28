from scipy.stats import bernoulli
from scipy.stats import uniform
from scipy.stats import norm
from scipy.stats import binom
import math
import random
import numpy as np
import scipy.stats as stats

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

class Node:

	def __init__(self,name):
		self.type = "Node"
		self.name = name
		self.params = {}
		self.children = []
		self.parents = []
		self.limits = []
	
	def set_params(self,params):
		for key, value in params.iteritems():
			self.params[key] = value
			
	def is_root(self):
		return len(self.parents) == 0
		
	def is_leaf(self):
		return len(self.children) == 0
		
	def add_parent(self,parent):
		assert isinstance(parent,Node)
		#add the parent to this node's parents
		#add this node to the parent's children
		self.parents.append(parent)
		parent.children.append(self)
		
	def add_child(self,child):
		assert isinstance(child,Node)
		#add the child to this node's children
		#add this node to the childs's parents
		self.children.append(children)
		child.parents.append(self)
		
class DiscreteNode(Node):
	def __init__(self,name=None,num_vals=2):	
		Node.__init__(self,name)
		# The number of values the discrete variable takes on
		self.num_vals = 2
		
class BernoulliNode(DiscreteNode):
	def __init__(self,name=None):
		DiscreteNode.__init__(self,name,2)
		
class GaussianNode(Node):
	def __init__(self,name):
		Node.__init__(self,name)
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
		
	
class LinearGaussianNode(Node):
	
	def __init__(self,name):
		Node.__init__(self,name)
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
		
