import math
import numpy as np 


def check_fit_condition(sample, token, position, except_this_token):
    if not except_this_token:
        if sample[position] == token:
            return True
        return False
    else:
        if sample[position] == token:
            return False
        return True

    
def get_sample_by_one_condition(text, token, position, except_this_token=False, N=500):
    """Get sample from real dataset, used by task: control_pos_content.
    
    Default text is seq2seq.decoder_in_test.
    Return index that sample[position] == token.
    If except_this_token == True: select all samples that sample[position] != token.
    The position is counted from 1. (Leave SOS in the front.)
    """
    assert position > 0, "Position is counted from 1."
    container = []
    
    for i, sample in enumerate(text):
        if check_fit_condition(sample, token, position, except_this_token): 
            container.append(i)
            if len(container) == N:
                break
    if except_this_token:
        print("\tFind %d samples with token!=%d and position=%d." % (len(container), token, position))
    else:
        print("\tFind %d samples with token==%d and position=%d." % (len(container), token, position))        
    return container
            
            
def get_same_amount_sample(samples):
	minimum_number = 1e6
	for sample in samples:
		if len(sample) < minimum_number:
			minimum_number = len(sample)  # From here 這邊後面還沒看過
	print("\tLeft %d data in each class." % minimum_number)
	#assert minimum_number > 0, "No any sample in some conditions."
	if minimum_number == 0:
		return
    
	for i, sample in enumerate(samples):
		samples[i] = samples[i][:minimum_number]
        
	return np.array(samples)


def truncate_toarray(list_, N, verbose=1):
    # Cut or pad list_ to specified length = N.
    arr = np.array(list_)
    assert len(arr.shape) == 1, "Only support 1d array."
    if arr.shape[0] < N:
        power = math.ceil(math.log(N / len(list_), 2))  # Repeat how many times.
        if verbose:
            print("N = %d, len = %d, power = %.1f" % (N, len(list_), power))
        for i in range(power):
            arr = np.concatenate([arr, arr])
    return arr[:N]
  
    
def get_different_amount_sample(samples, threshold=5):  
    # It is a slower way to deal with different amount, but it is faster for coding.
    # It will copy sample index, then copy hidden state, then classify those states.
    # Not sure which kind of sample is most or least.
    # Support multiple kinds of sample.
    least_sample_N = min([len(x) for x in samples])
    if least_sample_N < threshold:
        return np.empty([len(samples), 1])
    
    most_sample_N = max([len(x) for x in samples])
    container = np.empty([len(samples), most_sample_N], dtype=int)  # Index is type integer.
    for i, si in enumerate(samples):
        container[i] = truncate_toarray(si, most_sample_N)
    # print(len(si1), len(si2), most_sample_N, container.shape)
    return container

