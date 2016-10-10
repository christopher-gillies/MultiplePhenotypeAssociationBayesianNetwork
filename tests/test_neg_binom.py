from .context import mpabn
from mpabn import bayesian_network as bn
import numpy as np
from scipy import stats
import pandas as pd
from mpabn import helpers
from scipy.stats import nbinom
np.random.seed(0)

"""
py.test -s tests/test_neg_binom.py 
"""


def test_p_neg_binom():
	
	alpha = 1
	mean = 45
	x = 7
	log_prob = helpers.p_neg_binom(x,alpha,mean,log=True)
	
	var = mean + alpha * (mean ** 2)
	r = 1/np.float(alpha)
	assert type(r) is np.float
	p = r / (r + mean)
	
	exp_r = (mean ** 2) / (var - mean)
	np.testing.assert_almost_equal(r, exp_r)
	#logpmf(x, n, p, loc=0
	
	log_prob_exp = nbinom.logpmf(x,r,p)
	
	np.testing.assert_almost_equal(log_prob,log_prob_exp)
	
	
	

def test_r_neg_binom():
	
	alpha = 2
	mean = 45
	var = mean + alpha * (mean ** 2)
	sample = helpers.r_neg_binom(alpha,mean,num=1000)
	
	calc_mean = np.mean(sample)
	print mean
	print calc_mean
	
	assert np.abs(calc_mean - mean) < 5
	
	calc_var = np.var(sample,ddof=1)
	print np.sqrt(var)
	print np.sqrt(calc_var)
	
	assert np.abs(np.sqrt(var) - np.sqrt(calc_var)) < 5
	
	
def test_neg_binom_node_prob():
	x1 = bn.BinaryNode("X1")
	x2 = bn.BinaryNode("X2")
	x3 = bn.NegativeBinomialNode("X3")
	
	x1.set_params(0.5)
	x2.set_params(0.5)
	
	x3.add_parent(x1)
	x3.add_parent(x2)
	
	alpha = 2
	x3.set_params([5, 2, 1],2)
	

	np.testing.assert_almost_equal(x3.prob({"X1": 1, "X2": 1, "X3": 12}), helpers.p_neg_binom(x=12,alpha=alpha,mean=np.exp(5 + 2 + 1),log=True))
	
	np.testing.assert_almost_equal(x3.prob({"X1": 0, "X2": 1, "X3": 12}), helpers.p_neg_binom(x=12,alpha=alpha,mean=np.exp(5 + 1),log=True))
	
	
	np.testing.assert_almost_equal(x3.prob({"X1": 1, "X2": 0, "X3": 12}), helpers.p_neg_binom(x=12,alpha=alpha,mean=np.exp(5 + 2),log=True))
	
	
	
def test_neg_binom_node_simulate():
	x1 = bn.BinaryNode("X1")
	x2 = bn.BinaryNode("X2")
	x3 = bn.NegativeBinomialNode("X3")

	x1.set_params(0.5)
	x2.set_params(0.5)

	x3.add_parent(x1)
	x3.add_parent(x2)

	alpha = 2
	x3.set_params([5, 2, 1],2)
		
	network = bn.BayesianNetwork()
	network.set_nodes([x1,x2,x3])

	samples = network.forward_sample(2000)
	
	exp = 0.25 * np.exp(5) + 0.25 * np.exp(5 + 2)  + 0.25 * np.exp(5 + 1) + 0.25 * np.exp(5 + 2 + 1)
	
	print np.mean(samples["X1"])
	print np.mean(samples["X2"])
	print np.log(np.mean(samples["X3"]))
	print np.log(exp)
	assert np.abs(  np.log(np.mean(samples["X3"])) - np.log(exp)) < 2
	
	
		
	
def test_neg_binom_node_mle():
	x1 = bn.BinaryNode("X1")
	x3 = bn.NegativeBinomialNode("X3")

	x1.set_params(0.5)

	x3.add_parent(x1)

	alpha = 2
	beta0 = 5
	beta1 = 2
	x3.set_params([beta0, beta1],alpha)
		
	network = bn.BayesianNetwork()
	network.set_nodes([x1,x3])

	samples = network.forward_sample(1000)
	
	exp = 0.55 * np.exp(beta0) + 0.5 * np.exp(beta1)
	
	print np.mean(samples["X1"])
	print np.log(np.mean(samples["X3"]))
	print np.log(exp)
	assert np.abs(  np.log(np.mean(samples["X3"])) - np.log(exp)) < 2
	
	
	network.mle(samples)
	print x3.params
	