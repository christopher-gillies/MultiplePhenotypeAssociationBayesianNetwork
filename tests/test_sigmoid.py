from .context import mpabn
from mpabn import bayesian_network as bn
import numpy as np
from scipy import stats
import pandas as pd
np.random.seed(0)
from mpabn import helpers

"""
py.test -s tests/test_sigmoid.py 
"""
def test_prob():
	
	p1 = bn.DiscreteNode("X1",[0,1])
	p2 = bn.DiscreteNode("X2",[0,1])
	node = bn.SigmoidNode("X3",[0,1])
	print node
	node.add_parent(p1)
	node.add_parent(p2)
	node.set_params([0,1,2])
	
	p_x = np.exp(node.prob({"X1":1,"X2":1,"X3":1}))
	
	np.testing.assert_almost_equal(p_x,helpers.sigmoid(1 + 2))
	
	p_x_0 = np.exp(node.prob({"X1":1,"X2":1,"X3":0}))
	
	np.testing.assert_almost_equal(p_x_0,1 - helpers.sigmoid(1 + 2))


def test_prob_2():
	node = bn.SigmoidNode("X3",[0,1])
	node.set_params([1])
	p_x = np.exp(node.prob({"X3":1}))
	print helpers.sigmoid(1)
	np.testing.assert_almost_equal(p_x,helpers.sigmoid(1))
	
	
def test_simulate():
	node = bn.SigmoidNode("X3",[0,1])
	node.set_params([1])
	x = np.zeros(1000)
	d = {}
	for i in range(0,1000):
		x[i] = node.simulate(d)
	
	print np.mean(x)
	assert np.abs(np.mean(x) - helpers.sigmoid(1) < 0.05)