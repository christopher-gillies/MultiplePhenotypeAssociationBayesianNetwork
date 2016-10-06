from .context import mpabn
from mpabn import bayesian_network as bn
import numpy as np
from scipy import stats
import pandas as pd
np.random.seed(0)
from mpabn import helpers

"""
py.test -s tests/test_normal.py 
"""
def test_prob():
	
	node = bn.LinearGaussianNode("X1")
	#set intercept to b 1 and variance to be 1
	node.set_params([1],10)
	
	p_x = np.exp(node.prob({"X1":2}))
	
	np.testing.assert_almost_equal(p_x,stats.norm.pdf(2,loc=1,scale=10))
	
	

def test_prob():

	node = bn.LinearGaussianNode("X1")
	#set intercept to b 1 and variance to be 1
	node.set_params([1],10)

	p_x = np.exp(node.prob({"X1":2}))

	np.testing.assert_almost_equal(p_x,stats.norm.pdf(2,loc=1,scale=10))