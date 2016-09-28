import numpy as np
from .context import mpabn
from mpabn import bayesian_network as bn

def test_ex():
	print np.sqrt(2)
	assert np.sqrt(2) == np.sqrt(2)
	
def test_create_node():
	node = bn.Node("1")
	assert not node == None