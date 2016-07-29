import numpy as np


def file_string(filename):

	with open(filename, 'r') as f:

		return f.read()		


def data_and_labels(filenames):

	n   = 200
	cut = 100

	texts  = np.zeros((0,200)) 
	labels = []
	for no, filename in enumerate(filenames):

		f_str = file_string(filename)

		char_arr = np.array([ord(i) for i in f_str])
		char_arr = char_arr[:len(char_arr)//200 *200]
		char_arr = char_arr.reshape((len(char_arr)+1)//n, n)		
		#char_arr = char_arr[:cut]
		
		lab = [[no, int(not(no))] for i in range(char_arr.shape[0])]
		texts = np.vstack((texts, char_arr))
		labels.extend(lab)	

	for i, v in enumerate(np.unique(texts)):
    		texts[np.where(texts == v)] = i
				
	
	return texts, labels

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


