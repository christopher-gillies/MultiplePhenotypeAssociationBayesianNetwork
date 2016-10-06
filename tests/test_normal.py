from .context import mpabn
from mpabn import bayesian_network as bn
import numpy as np
from scipy import stats
import pandas as pd
from mpabn import helpers
from scipy.stats import norm

np.random.seed(0)

"""
py.test -s tests/test_normal.py 
"""
def test_prob():
	
	node = bn.LinearGaussianNode("X1")
	#set intercept to b 1 and variance to be 1
	node.set_params([1],10)
	
	p_x = np.exp(node.prob({"X1":2}))
	
	np.testing.assert_almost_equal(p_x,stats.norm.pdf(2,loc=1,scale=10))
	
	

def test_prob_2():
	x1 = bn.BinaryNode("X1")
	x1.set_params(0.5)
	x2 = bn.BinaryNode("X2")
	x2.set_params(0.5)
	x3 = bn.LinearGaussianNode("X3")
	x3.add_parent(x1)
	x3.add_parent(x2)
	#set intercept to b 1 and variance to be 1
	x3.set_params([1,2,3],2)

	p_x = np.exp(x3.prob({"X3":6,"X1":1,"X2":1}))

	np.testing.assert_almost_equal(p_x,stats.norm.pdf(6,loc=6,scale=2))
	
	

def test_simulate():
	x1 = bn.BinaryNode("X1")
	x1.set_params(0.5)
	x2 = bn.BinaryNode("X2")
	x2.set_params(0.5)
	x3 = bn.LinearGaussianNode("X3")
	x3.add_parent(x1)
	x3.add_parent(x2)
	#set intercept to b 1 and variance to be 1
	x3.set_params([1,2,3],2)
	
	n = 20000
	sample = np.zeros(n)
	for i in range(0,n):
		sample[i] = x3.simulate({"X1":1,"X2":1})
	
	print np.mean(sample)
	
	assert np.abs(np.mean(sample) - 6) < 0.02
	


def test_forward_sample_and_mle():
	x1 = bn.BinaryNode("X1")
	x1.set_params(0.5)
	x2 = bn.BinaryNode("X2")
	x2.set_params(0.5)
	x3 = bn.LinearGaussianNode("X3")
	x3.add_parent(x1)
	x3.add_parent(x2)
	#set intercept to b 1 and variance to be 1
	x3.set_params([1,2,3],2)

	network = bn.BayesianNetwork()
	network.set_nodes([x1,x2,x3])
	
	sample = network.forward_sample(10000)

	print np.mean(sample["X3"])

	assert np.abs(np.mean(sample["X3"]) - 14/4.0) < 0.05
	
	
	network.mle(sample)
	
	print x3.get_params()
	
	betas, std_dev =  x3.get_params()
	
	assert np.abs(betas[0] - 1) < 0.1
	assert np.abs(betas[1] - 2) < 0.1
	assert np.abs(betas[2] - 3) < 0.1
	assert np.abs(std_dev - 2) < 0.1
	
	
def test_gaussian_node():
	x1 = bn.GaussianNode("X1")
	x1.set_params(4.0,5.0)
	
	np.testing.assert_almost_equal(x1.prob({"X1":1},log=False),norm.pdf(1,loc=4,scale=5))
	np.testing.assert_almost_equal(x1.prob_easy(1),norm.pdf(1,loc=4,scale=5))