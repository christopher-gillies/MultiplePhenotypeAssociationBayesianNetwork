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
	
	n = 100
	sim_vals = np.zeros(n)
	for i in range(0,n):
		sim_vals[i] = x1.simulate({ "X2": [0], "X3": [1] })
	print np.mean(sim_vals)
	
	assert x1.prob({ "X1":1,"X2":1,"X3":0 }) == np.log(0.7)

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
	x1 = bn.DiscreteNode("X1",['n','y'])
	x2 = bn.DiscreteNode("X2",[0,1])
	x3 = bn.DiscreteNode("X3",[0,1])
	x1.add_parent(x2)
	x1.add_parent(x3)
	params = pd.DataFrame(
	[
		[0.1,'y',0,0],
		[0.9,'n',0,0],
		[0.2,'y',0,1],
		[0.8,'n',0,1],
		[0.3,'y',1,0],
		[0.7,'n',1,0],
		[0.4,'y',1,1],
		[0.6,'n',1,1]
	],columns=['prob','X1','X2','X3'])
	x1.set_params(params)
	assert isinstance(x1.params,pd.DataFrame)

	x2.set_params( 
	pd.DataFrame(
	[
		[0.1,0],
		[0.9,1]
	]
	,columns=['prob','X2'])
	)
	
	assert x2.prob({ "X2": 1}) == np.log(0.9)
	
	x3.set_params( 
	pd.DataFrame(
	[
		[0.8,0],
		[0.2,1]
	]
	,columns=['prob','X3'])
	)
	
	assert x3.prob({ "X3": 0}) == np.log(0.8)

	network.set_nodes([x1,x2,x3])
	network.print_network()
	
	#test joint
	np.testing.assert_almost_equal(network.joint_prob({ "X1": 'y', "X2":0, "X3":0 },log=False),  (0.1 * 0.8 * 0.1))
	#test log likelihood
	
	df_test = pd.DataFrame({ "X1": ['y','y'], "X2": [0,0], "X3": [0,0] })
	
	exp_llh = 2 * network.joint_prob({ "X1": 'y', "X2":0, "X3":0 })
	print exp_llh
	np.testing.assert_almost_equal(exp_llh, network.complete_data_log_likelihood(df_test))
	
	res = network.forward_sample(200)
	print res
	#print "Mean X1 {0}".format(np.mean(res["X1"]))
	print "Mean X2 {0}".format(np.mean(res["X2"]))
	print "Mean X3 {0}".format(np.mean(res["X3"]))
	#x1.mle(res)
	#x2.mle(res)
	#x3.mle(res)
	network.mle(res)
	print x1.params
	print x2.params
	print x3.params
	

def test_set_has_latent_descendant():
	print "Test has latent descendant"
	x1 = bn.DiscreteNode("X1",[0,1])
	x2 = bn.DiscreteNode("X2",[0,1])
	x3 = bn.DiscreteNode("X3",[0,1])
	x4 = bn.DiscreteNode("X4",[0,1])
	x5 = bn.DiscreteNode("X5",[0,1])
	
	x1.add_child(x2)
	x2.add_child(x3)
	x3.add_child(x4)
	x4.add_child(x5)
	
	x4.set_is_latent()
	
	nodes = [x1,x2,x3,x4,x5]
	network = bn.BayesianNetwork()
	network.set_nodes(nodes)
	network.print_network()
	
	assert x4.has_latent_descendant == False
	assert x5.has_latent_descendant == False
	
	assert x1.has_latent_descendant == True
	assert x2.has_latent_descendant == True
	assert x3.has_latent_descendant == True

"""
Create a network
Create a small dataset and check the probabilities
"""
def test__prob_x_given_others__():
	print "test__prob_x_given_others__"
	network = bn.BayesianNetwork()
	x1 = bn.DiscreteNode("X1",['n','y'])
	x2 = bn.DiscreteNode("X2",[0,1])
	x3 = bn.DiscreteNode("X3",[0,1])
	x1.add_parent(x2)
	x1.add_parent(x3)
	params = pd.DataFrame(
	[
		[0.1,'y',0,0],
		[0.9,'n',0,0],
		[0.2,'y',0,1],
		[0.8,'n',0,1],
		[0.3,'y',1,0],
		[0.7,'n',1,0],
		[0.4,'y',1,1],
		[0.6,'n',1,1]
	],columns=['prob','X1','X2','X3'])
	x1.set_params(params)
	x2.set_params( 
	pd.DataFrame(
	[
		[0.1,0],
		[0.9,1]
	]
	,columns=['prob','X2'])
	)
	x3.set_params( 
	pd.DataFrame(
	[
		[0.8,0],
		[0.2,1]
	]
	,columns=['prob','X3'])
	)
	network.set_nodes([x1,x2,x3])
	
	d = {"X1":'y',"X2":1,"X3":1}
	
	res = network.__prob_x_given_others__(d,target=x1)
	np.testing.assert_almost_equal(res['y'],np.log(0.4))
	np.testing.assert_almost_equal(res['n'],np.log(0.6))
	
	d2 = {"X1":'y',"X2":1,"X3":1}
	
	res = network.__prob_x_given_others__(d2,target=x2)
	joint_d2_x2_1 = 0.9 * 0.2 * 0.4
	joint_d2_x2_0 = 0.1 * 0.2 * 0.2
	normalizer = joint_d2_x2_1 + joint_d2_x2_0
	np.testing.assert_almost_equal(res[1],np.log( joint_d2_x2_1 / normalizer))
	np.testing.assert_almost_equal(res[0],np.log(joint_d2_x2_0 / normalizer))


def test__hard_e_step():
	print "test__hard_e_step"
	network = bn.BayesianNetwork()
	x1 = bn.DiscreteNode("X1",['n','y'])
	x1.set_is_latent()
	x2 = bn.DiscreteNode("X2",[0,1])
	x3 = bn.DiscreteNode("X3",[0,1])
	network.set_nodes([x1,x2,x3])
	#override default implementation
	network.__prob_x_given_others__ = lambda x: { 'y':0.1 , 'n':0.001 }
	df = pd.DataFrame( { "X1":['n','n'], "X2":[0,0],"X3":[1,1] })
	print df
	network.__hard_e_step__(df)
	print df
	for index,row in df.iterrows():
		assert row["X1"] == 'y'
	
def test_hard_em_logic():
	print "test_hard_em_logic"
	#override
	#__hard_e_step__(data)
	#mle(data)
	#complete_data_log_likelihood
	
	network = bn.BayesianNetwork()
	x1 = bn.DiscreteNode("X1",['n','y'])
	x1.set_is_latent()
	x2 = bn.DiscreteNode("X2",[0,1])
	x3 = bn.DiscreteNode("X3",[0,1])
	network.set_nodes([x1,x2,x3])
	#override default implementation
	network.__hard_e_step__ = lambda x: None
	network.mle = lambda x: None
	
	
	def llh_func(x):
		llh_func.counter += 1
		if llh_func.counter == 1:
			return -10
		elif llh_func.counter == 2:
			return -8
		elif llh_func.counter == 3:
			return -6
		elif llh_func.counter == 4:
			return -2
		else:
			return -2

	llh_func.counter = 0
	network.complete_data_log_likelihood = llh_func
	
	df = pd.DataFrame( { "X1":['n','n'], "X2":[0,0],"X3":[1,1] })
	res = network.hard_em(df)
	assert res["num_iter"] == 5



#####
# need to update mle step for this to work properly with discrete variables
#####
def test_hard_em():
	print "test_hard_em"
	"""
	Try to learn X1 | X2,X3
	X1 --> X2 --> X3
	X1 --> X3
	"""
	x1 = bn.DiscreteNode("X1",[0,1])
	x1.set_is_latent()
	x2 = bn.DiscreteNode("X2",[0,1])
	x3 = bn.DiscreteNode("X3",[0,1])
	
	x1.add_child(x2)
	x1.add_child(x3)
	x2.add_child(x3)
	
	x1.set_params( 
	pd.DataFrame(
	[
		[0.5,0],
		[0.5,1]
	]
	,columns=['prob','X1'])
	)
	
	x2_params = pd.DataFrame(
	[
		[0.4,0,0],
		[0.6,1,0],
		[0.4,0,1],
		[0.6,1,1],
	],columns=['prob','X2','X1'])
	x2.set_params(x2_params)
	
	x3_params = pd.DataFrame(
	[
		[0.3,0,0,0],
		[0.7,1,0,0],
		[0.3,0,0,1],
		[0.7,1,0,1],
		[0.3,0,1,0],
		[0.7,1,1,0],
		[0.3,0,1,1],
		[0.7,1,1,1],
	],columns=['prob','X3','X1','X2'])
	x3.set_params(x3_params)
	
	network = bn.BayesianNetwork()
	network.set_nodes([x1,x2,x3])
	
	#forward sample
	sample = network.forward_sample(200)
	print sample
	network.mle(sample)
	print x3.params
	sample_copy = sample.copy()
	
	#set initial data
	for index,row in sample_copy.iterrows():
		#data.set_value(index,self.latent_node.name,max_val)
		sample_copy.set_value(index,"X1",row["X2"])
	
	#run em
	#should just do MLE params
	#network.hard_em(sample)
		
	#check accuracy
	
	