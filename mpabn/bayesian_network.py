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
from scipy.misc import logsumexp
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
from helpers import find_max_key_val_in_dict
from helpers import sigmoid
from helpers import logistic
from helpers import p_neg_binom
from helpers import r_neg_binom
from helpers import fit_neg_binom
import sklearn.linear_model as model
from scipy.stats import bernoulli

from statsmodels.discrete.discrete_model import NegativeBinomial as nb_glm

class BayesianNetwork:
	def __init__(self):
		pass
		
	def set_nodes(self,nodes):
		"""
		set the nodes for this network
		* all nodes have to have a distinct name
		* at most 1 latent node is allowed
		"""
		assert type(nodes) == list
		#validate that all are nodes and all names are distinct
		num_is_latent = 0
		names = {}
		latent_node = None
		for node in nodes:
			assert isinstance(node,_Node)
			assert node.name != "prob"
			if node.is_latent:
				latent_node = node
				num_is_latent += 1
			
			assert not names.has_key(node.name), "Duplicate name: {0}".format(node.name)
			names[node.name] = 1
			
		assert num_is_latent <= 1, "Only one latent node allowed"

		#compute topological sort
		self.__dfs__(nodes)
		assert self.nodes is not None
		self.latent_node = latent_node
		self.set_has_latent_descendant()
		
		
	def set_has_latent_descendant(self):
		"""
		loop through the parents of the latent node if it exists
		mark each parent as having a latent ancestor
		mark any ancestors of these parents
		"""
		assert self.nodes is not None
		if self.latent_node is not None:
			for parent in self.latent_node.parents:
				self.mark_has_latent_descendant(parent)
	
	def mark_has_latent_descendant(self,node):
		assert node is not None
		node.set_has_latent_descendant()
		for parent in node.parents:
			self.mark_has_latent_descendant(parent)
	
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
		assert self.nodes is not None, "Please set the nodes of the network"
		for node in self.nodes:
			print str(node)
		
	#forward sample
	def forward_sample(self, n=1):
		assert self.nodes is not None
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
		assert self.nodes is not None
		assert type(data) == pd.DataFrame
		for node in self.nodes:
			node.mle(data)
	
	#compute joint probability
	def	joint_prob(self,dict_vals,log=True):
		assert self.nodes is not None
		log_joint = 0.0
		for node in self.nodes:
			#print "joint " + str(node)
			log_joint += node.prob(dict_vals,log=True)
		
		if log:
			return log_joint
		else:
			return np.exp(log_joint)
	
	def complete_data_log_likelihood(self,data):
		assert self.nodes is not None
		assert type(data) == pd.DataFrame
		llh = 0.0
		for index,row in data.iterrows():
			llh += self.joint_prob(row.to_dict())
		return llh
		
		
		
	#perform hard expectation maximization
	#only support binary for time being
	#will consider mcmc for normal
	def hard_em(self,data,max_iter=100,initialization_func=None):
		assert self.latent_node is not None
		assert self.nodes is not None
		assert type(data) == pd.DataFrame
		#only allow this algorithm when the latent node has no parents
		assert len(self.latent_node.parents) == 0
		
		if initialization_func is not None:
			#apply initialization function to the data
			pass
			
			
		#assume data has been completed already as the initialization is application specific
		#mle of parameters to get initial parameters
		num_iter = 1
		self.mle(data)
		previous_llh = -np.inf
		current_llh = self.complete_data_log_likelihood(data)
		llhs = [current_llh]
		while True:
			print "Iteration: {0}".format(num_iter)
			print "Previous LLH: {0}".format(previous_llh)
			print "Current LLH: {0}".format(current_llh)
			#hard e step
			self.__hard_e_step__(data)
			#hard m step
			self.mle(data)
			previous_llh = current_llh
			current_llh = self.complete_data_log_likelihood(data)
			num_iter += 1
			llhs.append(current_llh)
			#if our log likelihood does not improve then
			if current_llh <= previous_llh:
				break
		
		return { "num_iter": num_iter, "llhs": llhs }
		
	def __prob_x_given_others__(self,dict_data,target = None):
		"""
		Computes the log conditional probability of each value from the target variable given
		all other variables.
		Return: dictionary [value] --> log prob
		* assumes all other variables have been specified
		"""
		
		if target is None:
			target = self.latent_node
		
		#check that all variables have been specified
		for node in self.nodes:
			if node is not target:
				assert dict_data.has_key(node.name)
			
		old_x = dict_data[target.name]
		log_probs = list()
		log_joint_prob_dict = dict()
		
		#compute joint probs
		for x_val in target.values:
			dict_data[target.name] = x_val
			log_joint_prob = self.joint_prob(dict_data)
			log_probs.append(log_joint_prob)
			#print "log_joint_prob: {0}".format(log_joint_prob)
			#print "x_val: {0}".format(x_val)
			#print "log_joint_prob: {0}".format(log_joint_prob)
			log_joint_prob_dict[x_val] = log_joint_prob
		
		#normalize joint probs	
		normalizer = logsumexp(log_probs)
		
		log_cond_prob = dict()
		for x_val,log_joint_prob in log_joint_prob_dict.iteritems():
			log_conditional_prob = log_joint_prob - normalizer
			log_cond_prob[x_val] = log_conditional_prob
			
		#reset input dictionary	
		dict_data[target.name] = old_x	
		
		return log_cond_prob
			
	def __hard_e_step__(self,data):
		"""
		loop through each row of the data
		compute the prob of each value of latent variable
		assign each rows value to be the MAP estimator (value with maximum probability: mode)
		"""
		assert self.latent_node is not None
		assert self.latent_node.num_vals == 2
		
		for index,row in data.iterrows():
			log_cond_probs = self.__prob_x_given_others__(row.to_dict())
			max_val,max_log_prob = find_max_key_val_in_dict(log_cond_probs)
			
			#assign the row to be the value with the maximum probability
			data.set_value(index,self.latent_node.name,max_val)
			#row[self.latent_node.name] = max_val
		
#Note that the underscore makes this class private
class _Node:
	"""
	All children must provide an implementation for: set_params, mle, simulate, prob
	"""
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
		self.has_latent_descendant = False
	
	def set_is_latent(self):
		self.is_latent = True
	
	def set_has_latent_descendant(self):
		self.has_latent_descendant = True
	
	def set_params(self,params):
		assert params is not None
		self.params = params
	
	def get_params(self):
		return self.params
			
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
		return "Name: {0}, is_latent: {1}, Children: {2}".format(self.name, self.is_latent,str(self.children_names.keys()))
		
	def mle(self):
		raise Exception("Not supported")
	
	def prob(self,dict_vals,log=True):
		#make sure data has been input correctly
		assert self.params is not None
		assert dict_vals.has_key(self.name)
		for parent_name in self.parent_names.keys():
			assert dict_vals.has_key(parent_name)
	
	def simulate(self,parent_vals=None):
		assert type(parent_vals) is dict
		for key,val in parent_vals.iteritems():
			assert self.parent_names.has_key(key)
				
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
		_Node.prob(self,dict_vals)
		
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
	
	#TODO: Need to update this routine to account for unobserved events
	# this does not work in those cases
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
				print "parent_vals: {0}".format(parent_vals)
				if len(self.parents) > 1:
					assert len(parent_vals) == len(parent_names)
				else:
					parent_vals = [parent_vals]
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
			###
			# Add back missing params
			# enumerate all param combos
			# check
			###
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

class SigmoidNode(_Node):
	# TODO:  override set_params, mle, simulate, prob
	def __init__(self,name,values):
		assert name is not None
		assert 0 in values
		assert 1 in values
		
		_Node.__init__(self,name)
		# The number of values the discrete variable takes on
		self.num_vals = len(values)
		self.values = values
	
	def set_params(self,params):
		"""
		params are a list
		first param is intercept
		all others are betas corresponding to each parent
		"""
		assert params is not None
		assert type(params) is list
		assert len(params) == len(self.parents) + 1
		_Node.set_params(self,params)
		
	
	def prob(self,dict_vals,log=True,index=0):
		#check that dictionary has all values needed for calculation
		_Node.prob(self,dict_vals)
		
		val_of_node = dict_vals[self.name]
		assert val_of_node is 0 or val_of_node is 1
		
		vals = [1]
		for par in self.parents:
			par_val = dict_vals[par.name]
			if type(par_val) is list:
				vals.append(par_val[index])
			else:
				vals.append(par_val)
		
		linear_comb = np.inner(self.params,vals)
		p_val_is_1 = sigmoid(linear_comb)
		
		p_val_of_node = None
		
		if val_of_node == 1:
			p_val_of_node = p_val_is_1
		elif val_of_node == 0:
			p_val_of_node = 1 - p_val_is_1
		else:
			raise("Error: Node: {0}, Value: {1} Not supported".format(self.name,val_of_node))
		
		if log:
			return np.log(p_val_of_node)
		else:
			return p_val_of_node
	
	def simulate(self,parent_vals=None):
		#perform assumption checking
		_Node.simulate(self,parent_vals)
		parent_vals_copy = parent_vals.copy()
		
		parent_vals_copy[self.name] = 1
		p_1 = self.prob(parent_vals_copy,log=False)
		return bernoulli.rvs(p_1, size=1)[0]
	
	def mle(self,data):
		assert type(data) is pd.DataFrame
		parent_names = []
		for par in self.parents:
			assert par.name in data.columns
			parent_names.append(par.name)
		#get columns of interest
		data_sub = data[parent_names].values
		response = data[self.name].values
		
		if len(self.parents) > 0:
			glm = model.LogisticRegression(C=10 ** 12)
			glm.fit(X=data_sub,y=response)
		
			params = []
			params.append(glm.intercept_[0])
			for beta in glm.coef_[0]:
				params.append(beta)	
		
			SigmoidNode.set_params(self,params)
		else:
			X = [ ]
			y = [ ]
			for v in response.tolist():
				X.append([1])
				y.append(v)
			glm = model.LogisticRegression(fit_intercept=False,C=10 ** 12)
			glm.fit(X=X,y=y)
			params = []
			params.append(np.float(glm.coef_[0]))	
			#print params
			SigmoidNode.set_params(self,params)


class ParentNode():
	"""
	Class for nodes without children
	has some common functions for these types of nodes
	"""
	def add_parent(self):
		raise Exception("{0} cannot have parents".format(type(self)))
		
	def prob_easy(self,val,log=False):
		vals_dict = {}
		vals_dict[self.name] = val
		return self.prob( vals_dict,log)
		

class BinaryNode(SigmoidNode,ParentNode):
	"""
	This class is a binary parent node that functions similarly to discrete node
	but it uses a sigmoid node for its implementation
	"""
	def __init__(self,name):
		SigmoidNode.__init__(self,name,[0,1])
	
	def set_params(self,p):
		assert type(p) is np.float
		#inverse sigmoid function (logistic function)
		intercept = - np.log(1/p - 1)
		SigmoidNode.set_params(self,[intercept])
						
	
class LinearGaussianNode(_Node):
	
	def __init__(self,name):
		_Node.__init__(self,name)
	
	def set_tolerance(self,tol):
		assert tol is not None
		assert type(tol) is np.float
		self.tol = tol
		
	def set_params(self,params,std_dev):
		"""
		params are a list
		first param is intercept
		all others are betas corresponding to each parent
		std_dev is a separate parameter for the normal distribution
		"""
		assert params is not None
		assert std_dev is not None
			
		assert type(params) is list
		assert len(params) == len(self.parents) + 1
		_Node.set_params(self,params)
		self.std_dev = np.float(std_dev)
	
	def get_params(self):
		return (self.params,self.std_dev)
	
	def prob(self,dict_vals,log=True,index=0):
		#check that dictionary has all values needed for calculation
		_Node.prob(self,dict_vals)

		val_of_node = dict_vals[self.name]

		vals = [1]
		for par in self.parents:
			par_val = dict_vals[par.name]
			if type(par_val) is list:
				vals.append(par_val[index])
			else:
				vals.append(par_val)
		
		#mean
		linear_comb = np.inner(self.params,vals)
			
		"""
		normal distribution density function can give positive values if the standard deviation is less than 1/sqrt(2 * p)
		we can address this is three ways
		(1) Compute the probability of a particular value x as normal_cdf(x + tolerance) - normal_cdf(x - tolerance). (Initially I was thinking of using a tolerance of std_dev/2) #https://en.wikipedia.org/wiki/List_of_logarithmic_identities
		(2) Standardize the input variables so that the distribution follows a standard normal.
		(3) Throw an error if the standard deviation of a normal distribution is less than 1/sqrt(2 * pi)
		
		After talking with William this really should not even be and issue
		"""
		
		log_p_val_of_node = norm.logpdf(val_of_node,loc=linear_comb,scale=self.std_dev)
		if log:
			return log_p_val_of_node
		else:
			return np.exp(log_p_val_of_node)
		
	def simulate(self,parent_vals=None,index=0):
		#perform assumption checking
		_Node.simulate(self,parent_vals)
		parent_vals_copy = parent_vals.copy()
		
		#collect values
		vals = [1]
		for par in self.parents:
			par_val = parent_vals[par.name]
			if type(par_val) is list:
				vals.append(par_val[index])
			else:
				vals.append(par_val)
				
		#mean
		linear_comb = np.inner(self.params,vals)
		x = norm.rvs(loc=linear_comb,scale=self.std_dev,size=1)
		return x

	def mle(self,data):
		assert type(data) is pd.DataFrame
		parent_names = []
		for par in self.parents:
			assert par.name in data.columns
			parent_names.append(par.name)
		#get columns of interest
		data_sub = data[parent_names].values
		response = data[self.name].values
		
		#number of predictors + 1
		p = len(self.parents) + 1
		def std_dev_est(fit_model,X,y):
			sigma_2 = np.sum((fit_model.predict(X) - y) ** 2) / float(len(y) - p)
			return np.sqrt(sigma_2)
		
		if len(self.parents) > 0:
			glm = model.LinearRegression()
			glm.fit(X=data_sub,y=response)

			params = []
			params.append(glm.intercept_)
			for beta in glm.coef_:
				params.append(beta)	

			LinearGaussianNode.set_params(self,params,std_dev_est(glm,data_sub,response))
		else:
			X = [ ]
			y = [ ]
			for v in response.tolist():
				X.append([1])
				y.append(v)
			glm = model.LinearRegression(fit_intercept=False)
			glm.fit(X=X,y=y)
			params = []
			params.append(np.float(glm.coef_))	
			#print params
			LinearGaussianNode.set_params(self,params,std_dev_est(glm,X,y))
			
			
			
class GaussianNode(LinearGaussianNode,ParentNode):
	"""
	GaussianNode that serves as a normal distribution parent node
	it uses a LinearGaussianNode as its implementation
	"""
	def __init__(self,name):
		LinearGaussianNode.__init__(self,name)
	
	def set_params(self,mean,std_dev):
		assert type(mean) is np.float
		assert type(std_dev) is np.float
		
		LinearGaussianNode.set_params(self,[mean],std_dev)
		
		

class NegativeBinomialNode(_Node):

	def __init__(self,name):
		_Node.__init__(self,name)
		
		
	def set_params(self,params,alpha):
		"""
		params are a list
		first param is intercept
		all others are betas corresponding to each parent
		alpha is the dispersion parameter
		"""
		assert params is not None
		assert alpha is not None
				
		assert type(params) is list
		assert len(params) == len(self.parents) + 1
		_Node.set_params(self,params)
		self.alpha = np.float(alpha)
		
		
	def prob(self,dict_vals,log=True,index=0):
		#check that dictionary has all values needed for calculation
		_Node.prob(self,dict_vals)

		val_of_node = dict_vals[self.name]

		vals = [1]
		for par in self.parents:
			par_val = dict_vals[par.name]
			if type(par_val) is list:
				vals.append(par_val[index])
			else:
				vals.append(par_val)

		#mean
		linear_comb = np.inner(self.params,vals)
		mean = np.exp(linear_comb)
		log_p_val_of_node = p_neg_binom(val_of_node,alpha=self.alpha,mean=mean,log=True)
		if log:
			return log_p_val_of_node
		else:
			return np.exp(log_p_val_of_node)
			
	def simulate(self,parent_vals=None,index=0):
		#perform assumption checking
		_Node.simulate(self,parent_vals)
		parent_vals_copy = parent_vals.copy()

		#collect values
		vals = [1]
		for par in self.parents:
			par_val = parent_vals[par.name]
			if type(par_val) is list:
				vals.append(par_val[index])
			else:
				vals.append(par_val)

		#mean
		linear_comb = np.inner(self.params,vals)
		mean = np.exp(linear_comb)
		x = r_neg_binom(alpha=self.alpha,mean=mean,num=1)
		return x
		
		
	def mle(self,data):
		assert type(data) is pd.DataFrame
		parent_names = []
		for par in self.parents:
			assert par.name in data.columns
			parent_names.append(par.name)
		#get columns of interest
		#column order should match parental order
		data_sub = data[parent_names].values
		response = data[self.name].values.tolist()

		if len(self.parents) > 0:
			#fit neg binomial model
			
			X = [ ]
			y = [ ]
			
			for i in range(0,len(response)):
				x = data_sub[i]
				x_temp = [1]
				x_temp.extend(x)
				yval = np.int(response[i])
				X.append(x_temp)
				y.append(yval)
				print y[i]
				print X[i]
				
			
			print len(y)
			print len(X)	
			res = fit_neg_binom(y,X)

			params = []
			params.extend(res["params"])
			alpha = np.float(res["alpha"])
			NegativeBinomialNode.set_params(self,params,alpha)
		else:
			X = [ ]
			y = [ ]
			for v in response:
				X.append([1])
				y.append(v)
			
			#fit neg binomial model
			res = fit_neg_binom(y,X)

			params = []
			params.extend(res["params"])
			alpha = np.float(res["alpha"])
			NegativeBinomialNode.set_params(self,params,alpha)