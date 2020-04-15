import numpy as np 
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_evaluate_real(seq2seq, sample_index):
    real = seq2seq.decoder_out_test[sample_index]
    real = np.squeeze(real, axis=-1)
    return real


def get_default_sample_from_seq2seq(seq2seq, mode="test"):
    if mode == "test": 
        real = seq2seq.decoder_out_test  # An array with list of index.
        pred = seq2seq.inference_batch(seq2seq.encoder_in_test)  # An array with list of index.
    real = np.squeeze(real, axis=-1)
    pred = pad_sequences(pred, maxlen=seq2seq.tgt_max_len, padding='post', truncating='post')
    #pred = np.argmax(pred, axis=-1)
    print("\t", real.shape, pred.shape)
    return real, pred


def evaluate_autoencoder(real=None, pred=None, seq2seq=None, verbose=0, PAD_token=0, EOS_token=2):
	"""
    Evaluate if real == pred.
    Whole accuracy: Only when every token are same in the whole sequence --> correct.
    Each accuracy: Calculate accuracy based on number of tokens.
    Real and pred should be both 2D arrays.
    
    Real = [[1, 1, 1], [2, 2, 2]], Pred = [[1, 1, 0], [2, 2, 2]]
    whole_accuracy = 50%, each_accuracy = 83%.
    """
	if real is None or pred is None:
		real, pred = get_default_sample_from_seq2seq(seq2seq)
	assert len(real.shape) == 2, "Real must be 2d-array."
	assert len(pred.shape) == 2, "Pred must be 2d-array."
	assert len(real.shape) == 2, "Real must be 2d-array."
	assert len(pred.shape) == 2, "Pred must be 2d-array."
    
	# Evaluate accuracy of whole sequence.
	mask_PAD = 1 - np.equal(real, PAD_token)  # Label padding as zero in y_true
	error = 1 - np.equal(real, pred)  # shape = (N, max_length)
	error = error * mask_PAD  # 1 if true != pred and true is not PAD.
	any_error = np.sum(error, axis=-1)
	any_error = np.clip(any_error, 0., 1.)
	whole_accuracy = np.mean(1 - any_error)
                
	# Evaluate accuracy of each token.
	mask_PAD = 1 - np.equal(real, PAD_token)  # Shape = (N, max_length, 1)
	mask_PAD = mask_PAD.astype(float)
    
	error = 1 - np.equal(real, pred)  # shape = (N, max_length)
	error = error * mask_PAD
	error = error.astype(float)
	each_accuracy = np.mean(1. - np.sum(error, axis=-1) / np.sum(mask_PAD, axis=-1))
    
	if verbose:
		print("=" * 50)
		print(real[:3])
		print(pred[:3])
		print("=" * 50)
	print("\twhole accuracy=%.4f, each accuracy=%.4f" % (whole_accuracy, each_accuracy))
	
	return whole_accuracy, each_accuracy



def evaluate_autoencoder_at_time(real=None, pred=None, seq2seq=None, time_step=0, 
                                 verbose=1, PAD_token=0, EOS_token=2):
    """
    Used after verification methods (replace some hidden states in decoder) in our paper for "autoencoder".
    For example, when target token = "A" and target T = 5.
    Then this accuracy only calculate whether model outputs "A" at t = 5.
    
    Real and pred should both be 2D arrays.
    time step: count from 1, no encoder output = 0, match position count, match verify_decoder.
    """
    if real is None or pred is None:
        real, pred = get_default_sample_from_seq2seq(seq2seq)
    assert len(real.shape) == 2, "Real must be 2d-array."
    assert len(pred.shape) == 2, "Pred must be 2d-array."
    assert real.shape == pred.shape, "Shape of real and pred should be the same."
    assert time_step > 0, "Output count from 1."
    
    error = 1 - np.equal(real, pred)  # shape = (N, max_length)
    error = error.astype(float)
    mask_PAD = 1 - np.equal(real, PAD_token)  # Shape = (N, max_length, 1)
    mask_PAD = mask_PAD.astype(float)
    
    mask_before, mask_current, mask_after = np.zeros(real.shape), np.zeros(real.shape), np.zeros(real.shape) 
    mask_before[:, :time_step-1] += 1
    mask_current[:, time_step-1] += 1
    mask_after[:, time_step:] += 1
    mask_after = mask_after * mask_PAD
    #print("mask =", mask_before, "\n", mask_current, "\n", mask_after)
    assert np.sum(mask_current * mask_PAD) == np.sum(mask_current), "Some sequences are too short."""
    
    masks = [mask_before, mask_current, mask_after]
    #error_name = ["before", "at", "after"]
    whole_accuracy_list = []
    each_accuracy_list = []
    for i, mask in enumerate(masks):
        masked_error = error * mask
        each_accuracy = np.mean(1. - np.sum(masked_error, axis=-1) / np.sum(mask, axis=-1))
        any_error = np.sum(masked_error, axis=-1)
        any_error = np.clip(any_error, 0., 1.)
        whole_accuracy = np.mean(1 - any_error)
        whole_accuracy_list.append(whole_accuracy)
        each_accuracy_list.append(each_accuracy)
    
    if verbose == 1:
        #print("\tWhole accuracy at time_step=%d: %.4f, %.4f, %.4f." % (time_step, whole_accuracy_list[0], whole_accuracy_list[1], whole_accuracy_list[2]))
        print("\tEach accuracy at time_step=%d: %.4f, %.4f, %.4f." % (time_step, each_accuracy_list[0], each_accuracy_list[1], each_accuracy_list[2]))
    elif verbose == 2:  # Output in log file.
        print("(t=%d)\t%.4f\t%.4f\t%.4f" % (time_step, each_accuracy_list[0], each_accuracy_list[1], each_accuracy_list[2]))
    return each_accuracy_list[1]


def evaluate_autoencoder_token(real=None, pred=None, token=None,  
                                verbose=1, PAD_token=0, EOS_token=2):
    """
    Calculate the probability (accuracy) of outputting a token at each time step.
    This is used after replacing hidden states of "counter" neurons.
    """
    assert len(real.shape) == 2, "Real must be 2d-array."
    assert len(pred.shape) == 2, "Pred must be 2d-array."
    assert real.shape == pred.shape, "Shape of real and pred should be the same."
    
    error = 1 - np.equal(token, pred)  # shape = (N, max_length)
    error = error.astype(float)
    mask_PAD = 1 - np.equal(real, PAD_token)  # Shape = (N, max_length, 1)
    mask_PAD = mask_PAD.astype(float)
    mask_current = np.zeros(real.shape)
    
    each_accuracy_list = []
    for time_step in range(pred.shape[1]):
        mask_current *= 0
        mask_current[:, time_step] += 1
        mask = mask_current * mask_PAD
        masked_error = error * mask
        if np.any(np.sum(mask, axis=-1) == 0.0):
            break
        each_accuracy = np.mean(1. - np.sum(masked_error, axis=-1) / np.sum(mask, axis=-1))
        each_accuracy_list.append(each_accuracy)
    if verbose == 1:
        print("\tEach accuracy:", [round(x, 2) for x in each_accuracy_list])

def evaluate_autoencoder_last_step(real=None, pred=None, EOS_token=2):
    """
    Used for "autoenc-last".
    Calculate only if last word is correct or not.
    And the last word should appear at the correct time step in pred data.
    """
    assert len(real.shape) == 2, "Real must be 2d-array."
    assert len(pred.shape) == 2, "Pred must be 2d-array."
    assert real.shape == pred.shape, "Shape of real and pred should be the same."
    
    correct = 0
    for i in range(real.shape[0]):
        for j in range(real.shape[1]):
            if real[i][j] == EOS_token:
                if pred[i][j] == EOS_token and pred[i][j-1] == real[i][j-1]:
                    correct += 1
                break
    return correct / float(real.shape[0])
    

def evaluate_token_position(control, pred):
    """
    Used for "token-posi".
    Calculate only if pred[control:position] = control:token
    """
    assert len(control.shape) == 2, "Control must be 2d-array."
    assert len(pred.shape) == 2, "Pred must be 2d-array."
    
    correct = 0
    for i in range(control.shape[0]):
        target_token, target_position = control[i][0], control[0][1]
        if pred[i][target_position - 1] == target_token:
            correct += 1
    return correct / float(control.shape[0])


    
"""
def check_accuracy(self, check_list=["word"], N=1000, inputs=None, real=None, pred=None, n=1000):
		#Compute accuracy of different control signals.
		# TODO: Support other signals: "rhyme", "POS".
		if inputs is None and real is None:
			inputs = self.encoder_in_test[:N]
			real = self.decoder_out_test[:N]
		if pred is None:
			pred = self.inference_batch(inputs)
			pred = pad_sequences(pred, maxlen=real.shape[1], padding='post')
		
		if len(real.shape) == 2:
			real_3d = np.expand_dims(real, axis=-1)
		else:
			real_3d = real
		#print(real.shape, pred.shape)
		#print("Pred =", pred[:3])  # (N, length, vocab)
		#print("Real =", real_3d[:3, :, 0])  # (N, length, 1)
		
		pred_3d = to_categorical(pred, num_classes=self.tgt_token_size)

		for check_item in check_list:
			if check_item == "word":
				accuracy_tensor = utils.each_accuracy(real_3d[:n], pred_3d[:n])
				x = K.get_value(accuracy_tensor)
				print(x[:3])
				accuracy = np.mean(K.get_value(accuracy_tensor))
				print("Accuracy (word) = %.3f" % accuracy)

				accuracy_tensor = utils.whole_accuracy(real_3d[:n], pred_3d[:n])
				x = K.get_value(accuracy_tensor)
				print(x[:3])
				accuracy = np.mean(K.get_value(accuracy_tensor))
				print("Accuracy (whole) = %.3f" % accuracy)

			elif check_item == "length":
				accuracy_tensor = utils.accuracy_length(real_3d[:n], pred_3d[:n])
				accuracy = np.mean(K.get_value(accuracy_tensor))
				print("Accuracy (length) = %.4f" % accuracy)

			elif check_item == "rhyme":
				import rhyme_process
				real = np.squeeze(real_3d)
				accuracy = rhyme_process.check_rhyme_accuracy(real, pred, self.tgt_itoc)
				print("Accuracy (rhyme) = %.4f" % accuracy)

			elif check_item == "content":
				for i in range(N):
					pred_word = [self.tgt_itoc[word] for word in list(pred[i])]
					print("content:", pred_word)

			else:
				print("This item %s is not supported in accuracy check list." % check_item)
				continue



"""