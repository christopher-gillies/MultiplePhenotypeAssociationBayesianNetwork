import numpy as np
from scipy.stats import nbinom

def all_true(vec):
	for v in vec:
		if not v:
			return False
	return True


def find_max_key_val_in_dict(in_dict):
	"""
	loop through the keys in the dictionary
	and find the key that has the maximum value
	"""
	max_key = None
	max_val = -np.inf
	for key,val in in_dict.iteritems():
		if val >= max_val:
			max_val = val
			max_key = key
	return (max_key,max_val)
	

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logistic(p):
    assert p > 0
    return np.log(p / (1 - p))


def p_neg_binom(x,alpha,mean,log=True):
	"""
	returns probability of x when x is negative binomially distributed
	x is the val to get the prob of
	alpha is the dispersion parameter
	mean is the mean of the negative binomial distribution
	returns log by default
	"""
	r =  1.0 / np.float(alpha)
	p = r / (r + mean)
	log_p = nbinom.logpmf(x, r, p, loc=0)
	if log:
		return log_p
	else:
		return np.exp(log_p)
		

def r_neg_binom(alpha,mean,num=1):
	"""
	returns negatively binomially distributed data
	alpha is the dispersion parameter
	mean is the mean of the negative binomial distribution
	num is the number of random numbers to return
	"""
	r =  1.0 / np.float(alpha)
	p = r / (r + mean)
	return nbinom.rvs(r,p,size=num)