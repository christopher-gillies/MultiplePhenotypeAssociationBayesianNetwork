import numpy as np
import pandas as pd
from .context import mpabn
from mpabn import bayesian_network as bn
from mpabn import helpers

"""
py.test tests/test_basic.py -s
"""
def test_ex():
	print np.sqrt(2)
	assert np.sqrt(2) == np.sqrt(2)

def test_pandas():
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
	#example of how to filter df for a specific row
	ind = params[['X1','X2','X3']].isin({'X1':[0], 'X3':[1], 'X2':[0]}).apply(helpers.all_true,axis=1)
	np.testing.assert_almost_equal( params.loc[ind].iloc[0]['prob'],0.2)

def test_pandas_2():
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
	#example of how to filter df for a specific row
	ind = params[['X2','X3']].isin({'X3':[1], 'X2':[0]}).apply(helpers.all_true,axis=1)
	p = params.loc[ind]['prob'].tolist()
	r = (np.random.multinomial(1,p,size=1) == 1).tolist()[0]
	params.loc[ind]['X1'].loc[r].iloc[0]
	

def test_find_max():
	d = { 'x':np.log(0.1), 'y':np.log(0.01), 'z':np.log(0.9), 'a':np.log(0.001)}
	key,val = helpers.find_max_key_val_in_dict(d)
	print "key,val = {0},{1}".format(key,val)
	assert key == 'z'
	np.testing.assert_almost_equal(val,np.log(0.9))
	
	
