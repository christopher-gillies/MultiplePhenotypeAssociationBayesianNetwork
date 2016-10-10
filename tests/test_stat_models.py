import numpy as np
import statsmodels.api as sm
from scipy.stats import nbinom
import scipy.special as sp
from scipy.misc import factorial
from statsmodels.discrete.discrete_model import NegativeBinomial as nb
from mpabn import helpers
"""
py.test tests/test_stat_models.py -s


"""

np.random.seed(0)

def p_nb(k,r,m):
	"""
	k is the number of observations
	r = 1 / dispersion
	m = mean
	"""
	t1 = sp.gamma(r + k) / ( factorial(k) * sp.gamma(r) )
	t2 = (m / (r + m)) ** k
	t3 = (r / (r + m)) ** r
	return t1 * t2 * t3

def l_p_nb(k,r,m):
	
	return sp.gammaln(r + k) - sp.gammaln(k + 1) - sp.gammaln(r) + k * (np.log(m) - np.log(r + m)) + r * (np.log(r) - np.log(r + m))	
	
#http://statsmodels.sourceforge.net/devel/index.html
def test_ols():
	# Generate artificial data (2 regressors + constant)
	nobs = 100
	X = np.random.random((nobs, 2))
	X = sm.add_constant(X)
	beta = [1, .1, .5]
	e = np.random.random(nobs)
	y = np.dot(X, beta) + e

	# Fit regression model
	results = sm.OLS(y, X).fit()

	# Inspect the results
	print results.summary()


def test_gamma_glm():
	data = sm.datasets.scotland.load()
	data.exog = sm.add_constant(data.exog)

	# Instantiate a gamma family model with the default link function.
	gamma_model = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())
	gamma_results = gamma_model.fit()
	
	# Inspect the results
	print gamma_results.summary()
	
	
def test_neg_bin():
	print "test neg binom"
	r = 0.4 # n in scipy stats nomenclature
	p = 0.7
	mean, var = nbinom.stats(r, p, moments='mv')
	print mean
	print var
	
	r_est =  mean ** 2 / (var - mean)
	np.testing.assert_almost_equal(r_est,r)
	
	p_est = r_est / (r_est + mean)
	np.testing.assert_almost_equal(p_est,p)
	
	
	sample = nbinom.rvs(r,p,size=1000)
	print np.mean(sample)
	print np.var(sample)
	

def test_neg_bin_2():
	"""
	simple regression test
	X ~ bernoulli(0.5)
	y ~ nb(mean = 2 + 2X, dispersion = 0.2)
	"""
	print "test neg binom 2"
	
	#1/dispersion parameter
	r = 100.0
	mean = 20
	#var = mean + 1/r * (mean ** 2)
	#p1 = r / np.float(r + mean)
	
	#mean_ex, var_ex = nbinom.stats(r, p1, moments='mv')
	#print mean
	#print mean_ex
	#print var
	#print var_ex
	
	n = 500
	#s1 = nbinom.rvs(r,p1,size=1000)
	s1 = helpers.r_neg_binom(alpha=1/r,mean=mean,num=n)
	print np.mean(s1)
	print np.var(s1)
	
	#assume same 1/dispersion but shift mean by 2
	mean = 30
	#var = mean + 1/r * (mean ** 2)
	#p2 = r / np.float(r + mean)
	
	s2 = helpers.r_neg_binom(alpha=1/r,mean=mean,num=n)
	print np.mean(s2)
	print np.var(s2)
	
	#print s2
	
	y = []
	y.extend(s1)
	y.extend(s2)
	
	x = []
	x.extend(np.zeros(n))
	x.extend(np.ones(n))
	
	#add intercept
	X = []
	for i in range(0,len(x)):
		X.append([1,np.int(x[i])])

		
	assert len(y) == 2 * n
	assert len(x) == 2 * n
	assert len(X) == 2 * n
	# Instantiate a gamma family model with the default link function.
	#default is that alpha is 1
	#nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
	#nb_results = nb_model.fit()
	
	nb_model_2 = nb(y,X)
	nb_results = nb_model_2.fit(maxiter=100)
	
	# Inspect the results
	print nb_results.summary()
	
	print nb_results.params
	
	# 1/r is the dispersion = scale
	# r = 1/ scale
	print 1/nb_results.params[-1]
	
	llh = 0.0
	llh2 = 0
	for i in range(0,(2 * n)):
		b = X[i]
		y_val = y[i]
		#last param is alpha
		#first two are betas
		mean_pred = np.exp( np.inner(b,nb_results.params[:-1]))
		np.testing.assert_almost_equal(nb_results.predict(b), mean_pred)
		#r_pred = 1/nb_results.scale
		r_pred = 1/nb_results.params[-1]
		#r_pred = 1 # for some reason this gives the same LLH
		p_pred = r_pred / (r_pred + mean_pred)
		#log_prob2 = nbinom.logpmf(y_val, r_pred, p_pred, loc=0)
		log_prob2 = helpers.p_neg_binom(y_val,nb_results.params[-1],mean_pred)
		llh2 += log_prob2
		log_prob = l_p_nb(y_val,r_pred,mean_pred)
		llh += log_prob
	
	print nb_results.llf
	print llh
	print llh2
	
	np.testing.assert_almost_equal(nb_results.llf,llh)
	np.testing.assert_almost_equal(llh2,llh)

def test_neg_bin_eq():
	k = 3
	r = 20
	mean = 4
	var = mean + 1/r * (mean ** 2)
	p = r / np.float(r + mean)
	np.testing.assert_almost_equal(nbinom.pmf(k,r,p), np.exp(l_p_nb(k,r,mean)))
	


def test_neg_bin_4():
	"""
	simple regression test
	X ~ bernoulli(0.5)
	y ~ nb(mean = 2 + 2X, dispersion = 0.2)
	"""
	print "test neg binom 4"

	#1/dispersion parameter
	r = 2.0
	mean = 100
	#var = mean + 1/r * (mean ** 2)
	#p1 = r / np.float(r + mean)

	#mean_ex, var_ex = nbinom.stats(r, p1, moments='mv')
	#print mean
	#print mean_ex
	#print var
	#print var_ex

	n = 500
	#s1 = nbinom.rvs(r,p1,size=1000)
	s1 = helpers.r_neg_binom(alpha=1/r,mean=mean,num=n)
	print np.mean(s1)
	print np.var(s1)

	#assume same 1/dispersion but shift mean by 2
	mean = 1000
	#var = mean + 1/r * (mean ** 2)
	#p2 = r / np.float(r + mean)

	s2 = helpers.r_neg_binom(alpha=1/r,mean=mean,num=n)
	print np.mean(s2)
	print np.var(s2)

	#print s2

	y = []
	y.extend(s1)
	y.extend(s2)

	x = []
	x.extend(np.zeros(n))
	x.extend(np.ones(n))

	#add intercept
	X = []
	for i in range(0,len(x)):
		X.append([1,np.int(x[i])])

	assert len(y) == 2 * n
	assert len(x) == 2 * n
	assert len(X) == 2 * n
	# Instantiate a gamma family model with the default link function.
	#default is that alpha is 1
	#nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
	#nb_results = nb_model.fit()

	nb_model_2 = sm.GLM(y, X, family=sm.families.NegativeBinomial())
	nb_results = nb_model_2.fit()
	# Inspect the results
	print nb_results.summary()
	
	alpha = nb_results.scale
	
	while(abs(nb_results.scale - 1) > 0.001):
		nb_model_2 = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha))
		nb_results = nb_model_2.fit()
		# Inspect the results
		print nb_results.summary()
		alpha = alpha * nb_results.scale
	

	print nb_results.params
	print 1/alpha
	
	
def test_neg_bin_5():

	print "test neg binom 5"

	#1/dispersion parameter
	r = 0.01
	mean = 100

	n = 500
	#s1 = nbinom.rvs(r,p1,size=1000)
	s1 = helpers.r_neg_binom(alpha=1/r,mean=mean,num=n)
	print np.mean(s1)
	print np.var(s1)

	#assume same 1/dispersion but shift mean by 2
	mean = 1000
	#var = mean + 1/r * (mean ** 2)
	#p2 = r / np.float(r + mean)

	s2 = helpers.r_neg_binom(alpha=1/r,mean=mean,num=n)
	print np.mean(s2)
	print np.var(s2)

	#print s2

	y = []
	y.extend(s1)
	y.extend(s2)

	x = []
	x.extend(np.zeros(n))
	x.extend(np.ones(n))

	#add intercept
	X = []
	for i in range(0,len(x)):
		X.append([1,np.int(x[i])])


	assert len(y) == 2 * n
	assert len(x) == 2 * n
	assert len(X) == 2 * n
	
	res = helpers.fit_neg_binom(y,X)
	
	print res["params"]
	print res["alpha"]
	print res["iter"]
	
	print res["results"].summary()