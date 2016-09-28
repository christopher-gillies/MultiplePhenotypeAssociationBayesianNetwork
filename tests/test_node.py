from .context import mpabn
from mpabn import bayesian_network as bn
import numpy as np
from scipy import stats

np.random.seed(0)
	
def test_create_node():
	node = bn.Node("1")
	assert node.is_root()
	assert node.is_leaf()
	
	
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