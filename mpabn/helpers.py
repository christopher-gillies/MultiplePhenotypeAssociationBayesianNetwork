import numpy as np

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