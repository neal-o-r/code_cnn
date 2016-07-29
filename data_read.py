import numpy as np


def file_string(filename):

	with open(filename, 'r') as f:

		return f.read()

		

def data_and_labels(filenames):

	n   = 200
	cut = 100

	texts  = np.zeros((0,200)) 
	labels = np.zeros(0)
	for no, filename in enumerate(filenames):

		f_str = file_string(filename)

		char_arr = np.array([ord(i) for i in f_str])
		char_arr = char_arr[:len(char_arr)//200 *200]
		char_arr = char_arr.reshape((len(char_arr)+1)//n, n)		
		#char_arr = char_arr[:cut]
		
		lab = np.zeros(char_arr.shape[0]) + no

		texts = np.vstack((texts, char_arr))
		labels = np.concatenate((labels, lab))	
		

	return texts, labels
