import numpy as np
import os

def file_string(filename):

	with open(filename, 'r') as f:

		return f.read()		


def data_and_labels():

	filenames = filter(lambda x: not x.startswith('.'), 
	            filter(os.path.isfile, ['data/'+i for i in os.listdir('data')])) 
	# yo dawg i heard you like filter...
	
	n   = 200 # block size 
	cut = 100 # optional thinning parameter

	texts  = np.zeros((0,200)) 
	labels = []
	for no, filename in enumerate(filenames):

		f_str = file_string(filename)

		char_arr = np.array([ord(i) for i in f_str])
		char_arr = char_arr[:len(char_arr)//200 *200]
		char_arr = char_arr.reshape((len(char_arr)+1)//n, n)		
		#char_arr = char_arr[:cut]
		
		lab = [0]*len(filenames)
		lab[no] = 1
		lab = [lab] * char_arr.shape[0]
		texts = np.vstack((texts, char_arr))
		labels.extend(lab)	

	for i, v in enumerate(np.unique(texts)):
    		texts[np.where(texts == v)] = i
					
	return texts, labels

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
	shuffle_indices = np.random.permutation(np.arange(data_size))
     	shuffled_data = data[shuffle_indices]
        
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
