from .context import mpabn
from mpabn import bayesian_network as bn
import numpy as np
from scipy import stats
import pandas as pd
np.random.seed(0)
	
	
	
def test_create_lg_node():
	print "Testing LinearGaussianNode"
	lg_node = bn.LinearGaussianNode("X")
	assert lg_node.is_root()
	assert lg_node.is_leaf()
	assert len(lg_node.children) == 0
	
	

def test_create_gaussian_node():
	node = bn.GaussianNode("1")
	node.set_mean(1)
	node.set_std(2)
	sample = node.simulate(100)
	assert(len(sample) == 100)
	
	sample_2 = stats.norm.rvs(size=100, loc=1, scale=2)
	
	stat, pval = stats.ks_2samp(sample, sample)
	print "P-value from ks-test: {0}".format(pval)
	assert pval > 0.05
	
def test_create_discrete_node_1():
	print "Testing DiscreteNode 1"
	node = bn.DiscreteNode("X1",[0,1])
	assert node.is_root()
	assert node.is_leaf()
	assert len(node.children) == 0
	assert len(node.values) == 2
	
	params = pd.DataFrame({ 
	"prob" : [0.3,0.7], 
	"X1": [0,1]})
	node.set_params(params)
	assert isinstance(node.params,pd.DataFrame)
	
	sim_vals = []
	for i in range(0,100):
		sim_vals.append(node.simulate())
	print np.mean(sim_vals)
	
def test_create_discrete_node_2():
	print "Testing DiscreteNode 2"
	x1 = bn.DiscreteNode("X1",[0,1])
	x2 = bn.DiscreteNode("X2",[0,1])
	x3 = bn.DiscreteNode("X3",[0,1])
	x1.add_parent(x2)
	x1.add_parent(x3)
	params = pd.DataFrame(
	[
		[0.1,0,0,0],
		[0.9,1,0,0],
		[0.2,0,0,1],
		[0.8,1,0,1],
		[0.3,0,1,0],
		[0.7,1,1,0],
		[0.4,0,1,1],
		[0.6,1,1,1]
	],columns=['prob','X1','X2','X3'])
	x1.set_params(params)
	assert isinstance(x1.params,pd.DataFrame)
	
	sim_vals = []
	for i in range(0,100):
		sim_vals.append(x1.simulate({ "X2": [0], "X3": [1] }))
	print np.mean(sim_vals)
	

def test_dfs_1():
	network = bn.BayesianNetwork()
	x1 = bn.DiscreteNode("X1",[0,1])
	x2 = bn.DiscreteNode("X2",[0,1])
	x3 = bn.DiscreteNode("X3",[0,1])
	x4 = bn.DiscreteNode("X4",[0,1])
	
	x4.add_parent(x1)
	x4.add_parent(x2)
	x4.add_parent(x3)
	
	x2.add_parent(x1)
	x1.add_parent(x3)
	
	network.set_nodes([x1,x2,x3,x4])
	network.print_network()

def test_forward_sample_1():
	print "Testing forward sample"
	network = bn.BayesianNetwork()
	x1 = bn.DiscreteNode("X1",[0,1])
	x2 = bn.DiscreteNode("X2",[0,1])
	x3 = bn.DiscreteNode("X3",[0,1])
	x1.add_parent(x2)
	x1.add_parent(x3)
	params = pd.DataFrame(
	[
		[0.1,0,0,0],
		[0.9,1,0,0],
		[0.2,0,0,1],
		[0.8,1,0,1],
		[0.3,0,1,0],
		[0.7,1,1,0],
		[0.4,0,1,1],
		[0.6,1,1,1]
	],columns=['prob','X1','X2','X3'])
	x1.set_params(params)
	assert isinstance(x1.params,pd.DataFrame)

	x2.set_params( 
	pd.DataFrame(
	[
		[0.5,0],
		[0.5,1]
	]
	,columns=['prob','X2'])
	)
	
	x3.set_params( 
	pd.DataFrame(
	[
		[0.5,0],
		[0.5,1]
	]
	,columns=['prob','X3'])
	)
	
	network.set_nodes([x1,x2,x3])
	network.print_network()
	print network.forward_sample()