import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K


def masked_perplexity_loss(y_true, y_pred, PAD_token=0):
	"""Construct customer masked perplexity loss."""
	mask = K.all(K.equal(y_true, PAD_token), axis=-1)  # Label padding as zero in y_true
	mask = 1 - K.cast(mask, K.floatx())
	nomask = K.sum(mask)
	loss = K.sparse_categorical_crossentropy(y_true, y_pred) * mask  # Multiply categorical_crossentropy with the mask
	print("loss =", loss.shape)
	return tf.exp(K.sum(loss)/nomask)



def whole_accuracy(real, pred, PAD_token=0):
	"""Correct when the whole sequence are the same, excluding after EOS."""
	real = K.cast(K.squeeze(real, axis=-1), K.floatx())  # Remove dim.shape = 1.
	pred = K.cast(K.argmax(pred, axis=-1), K.floatx())
    
	mask_PAD = K.equal(real, PAD_token)  # Label padding as zero in y_true
	mask_PAD = 1 - K.cast(mask_PAD, K.floatx())

	error = 1 - K.cast(K.equal(real, pred), K.floatx())  # shape = (N, max_length)
	error = error * mask_PAD  # 1 if true != pred and true is not PAD.
	any_error = K.sum(error, axis=-1)
	any_error = K.clip(any_error, 0., 1.)
	accuracy = 1 - any_error

	return accuracy 


def each_accuracy(real, pred, PAD_token=0):
	"""Count accuracy based on each word."""
	# Use in training metrics.
	real = K.cast(K.squeeze(real, axis=-1), K.floatx())  # Remove dim.shape = 1.
	pred = K.cast(K.argmax(pred, axis=-1), K.floatx())
    
	mask_PAD = K.equal(real, PAD_token) 
	mask_PAD = 1 - K.cast(mask_PAD, K.floatx())

	error_tensor = 1 - K.cast(K.equal(real, pred), K.floatx())  # shape = (N, max_length)
	error_tensor = error_tensor * mask_PAD

	accuracy = 1 - K.sum(error_tensor, axis=-1) / K.sum(mask_PAD, axis=-1)
	return accuracy  


def accuracy_length(y_true, y_pred):
	"""Compute the length control signal accuracy by matching position of EOS token."""

	EOS_token = 2
	
	mask_PAD = K.all(K.equal(y_true, 0), axis=-1)  # Shape = (N, max_length, 1)
	mask_PAD = 1 - K.cast(mask_PAD, K.floatx())
	mask_PAD = tf.squeeze(mask_PAD)  # Shape = (N, max_length). False if this is PAD.

	y_pred = K.cast(K.argmax(y_pred, axis=-1), K.floatx())
	y_pred = y_pred * mask_PAD  # Shape = (N, max_length)

	filter_EOS = K.all(K.equal(y_true, EOS_token), axis=-1)
	filter_EOS = K.cast(filter_EOS, K.floatx())
	filter_EOS = tf.squeeze(filter_EOS)  # Shape = (N, max_length), True if it is EOS.

	y_expected = K.equal(y_pred * filter_EOS, float(EOS_token))
	y_expected = K.cast(y_expected, K.floatx())  # Shape = (N, max_length)
	y_expected = K.sum(y_expected, axis=-1)  # Shape = (N, )
	y_expected = K.cast((K.equal(y_expected, 1.0)), K.floatx())
	y_result = K.cast(K.equal(y_pred, float(EOS_token)), K.floatx())  # Shape = (N, max_length)
	y_result = K.sum(y_result, axis=-1)  # Shape = (N, )
	y_result = K.cast((K.equal(y_result, 1.0)), K.floatx())

	accuracy = y_expected * y_result  # Shape = (N, )
	accuracy = (K.sum(accuracy) / K.sum(filter_EOS))  # / K.sum(tf.ones_like(accu)))

	return accuracy


def get_trained_model(task='autoencoder', data_name='autoencoder', units=256, random_seed=2):
    import evaluator
    from seq2seq import Seq2Seq
    from argparse import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args(args=[])
    args.mode = "analysis"
    args.task = task
    args.data_name = data_name
    args.units = units
    args.random_seed = random_seed
    args.model_path = "../saved_model/%s_units=%s_seed=%d" % (
        args.data_name, args.units, args.random_seed)
    
    seq2seq = Seq2Seq(args)
    seq2seq.load_seq2seq(args.model_path)
    print("\tmode=%s, units=%d, model_path=%s" % (seq2seq.mode, seq2seq.units, seq2seq.model_path))
    if task == 'autoencoder' or task == 'autoenc-last':
        whole_accuracy, each_accuracy = evaluator.evaluate_autoencoder(seq2seq=seq2seq)
    else:
        raise 'no this task'
    assert whole_accuracy > 0.93 and each_accuracy > 0.99, "Load model failed."
    return seq2seq
    
    
