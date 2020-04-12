"""Get any state in trained model."""


#import data_reader
import numpy as np 
import pdb
import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K

from collections import defaultdict
from tensorflow.keras.activations import hard_sigmoid
from tensorflow.keras.activations import tanh
from tensorflow.keras.models import Model


# def sample_index_to_sample(text, sample_index):
    

def get_hidden_state(seq2seq, sample_index, SOS_token=1):
    # Dimension of hidden state = class, sample, length, dimension.
	container = np.empty([sample_index.shape[0], sample_index.shape[1], 
							seq2seq.tgt_max_len, seq2seq.units])
	dec_layer_model = Model(inputs=seq2seq.decoder_model.input, 
							outputs=seq2seq.decoder_model.get_layer('decoder_gru').get_output_at(-1))
    
	for i in range(sample_index.shape[0]): # Store each class in for loop.
		sample = seq2seq.encoder_in_test[sample_index[i]]
		encoder_output = seq2seq.encoder_model.predict(sample, batch_size=256)[0] 
		container[i, :, 0:1, :] = np.expand_dims(encoder_output, axis=1)

		decoder_states = encoder_output
		decoder_inputs = np.full([sample_index.shape[1], 1], SOS_token) # first token is SOS
		for t in range(seq2seq.tgt_max_len - 1):
			hidden = dec_layer_model.predict([decoder_inputs] + [decoder_states], verbose=0, batch_size=256)
			output_tokens, decoder_states = seq2seq.decoder_model.predict([decoder_inputs] + [decoder_states], verbose=0, batch_size=256)
			sampled_token_index = np.argmax(output_tokens[:, -1, :], axis=-1)  # Sample a token
			decoder_inputs[:, 0] = sampled_token_index
			container[i, :, t+1:t+2, :] = hidden[0]
	#container = container.transpose([0, 2, 1, 3])
	print("\t(Hidden value) container.shape =", container.shape)
	return container


def get_encoder_hidden_state(seq2seq, sample_index, SOS_token=1):
	"""Get each hidden state from encoder."""
	container = np.empty([sample_index.shape[0], sample_index.shape[1], seq2seq.src_max_len, seq2seq.units])

	# Store each class in for loop.
	for i in range(sample_index.shape[0]):
		sample = seq2seq.encoder_in_test[sample_index[i]]
		encoder_output = seq2seq.encoder_model.predict(sample)[1]  # (N, time_step, units)
		container[i] = np.copy(encoder_output)
	print("\t(Hidden value) container.shape =", container.shape)
	return container


def get_dense_values(seq2seq, sample_index, SOS_token=1):
	"""Get hidden values in last fully connected layer."""
	container = np.zeros([sample_index.shape[0], sample_index.shape[1], seq2seq.tgt_max_len, seq2seq.tgt_token_size])  # (12, 13, 500, 5000x)
	dense_layer = Model(inputs=seq2seq.decoder_model.input, 
						outputs=seq2seq.decoder_model.get_layer('output_dense').get_output_at(-1))
	
	for i in range(sample_index.shape[0]):
		sample = seq2seq.encoder_in_test[sample_index[i]]
		encoder_output = seq2seq.encoder_model.predict(sample, batch_size=256)[0] 
		decoder_states = encoder_output
		decoder_inputs = np.full([sample_index.shape[1], 1], SOS_token)  # The first token is SOS.

		for t in range(seq2seq.tgt_max_len):
			one_dense_values = dense_layer.predict([decoder_inputs] + [decoder_states], verbose=0, batch_size=256)
			output_tokens, decoder_states = seq2seq.decoder_model.predict([decoder_inputs] + [decoder_states], verbose=0, batch_size=256)
			sampled_token_index = np.argmax(output_tokens[:, -1, :], axis=-1)  # Sample a token.
			decoder_inputs[:, 0] = sampled_token_index[:]
			container[i, :, t] = np.copy(np.squeeze(one_dense_values, axis=1))  # Change shape: (500, 1, 5000x) -> (500, 5000x)

	return container


def get_gate_values(seq2seq, sample_index, SOS_token=1):
	"""Load gate values in GRU from trained Seq2Seq.""" # TODO: Change to read variable values from other way.

	def propagate_gru(weight, inputs, states, units):
		kernel = K.variable(weight[0])  # shape = (input_dim, self.units * 3)
		recurrent_kernel = K.variable(weight[1])  # shape = (self.units, self.units * 3)
		bias = K.variable(weight[2])  # bias_shape = (3 * self.units,)
		
		# Update gate.
		kernel_z = kernel[:, :units]
		recurrent_kernel_z = recurrent_kernel[:, :units]
		# Reset gate.
		kernel_r = kernel[:, units:units * 2]
		recurrent_kernel_r = recurrent_kernel[:, units:units * 2]
		# New gate.
		kernel_h = kernel[:, units * 2:]
		recurrent_kernel_h = recurrent_kernel[:, units * 2:]
		# Assume use bias, not reset_after
		input_bias_z = bias[:units]
		input_bias_r = bias[units: units * 2]
		input_bias_h = bias[units * 2:]
		# Bias for hidden state - just for compatibility with CuDNN.
		
		# Call 
		inputs = K.variable(inputs)	 # Not sure.
		h_tm1 = K.variable(states)	 # Not sure. Previous memory state.
		# Assume no dropout in this layer and self.implementation = 1 and not reset_after.
		x_z = K.bias_add(K.dot(inputs, kernel_z), input_bias_z)
		x_r = K.bias_add(K.dot(inputs, kernel_r), input_bias_r)
		x_h = K.bias_add(K.dot(inputs, kernel_h), input_bias_h)
				   
		recurrent_z = K.dot(h_tm1, recurrent_kernel_z)
		recurrent_r = K.dot(h_tm1, recurrent_kernel_r)
					
		z = hard_sigmoid(x_z + recurrent_z)  # Recurrent activation = 'hard_sigmoid'.
		r = hard_sigmoid(x_r + recurrent_r)

		recurrent_h = K.dot(r * h_tm1, recurrent_kernel_h)
		hh = tanh(x_h + recurrent_h)  # Activation = 'tanh'.	   
		h = z * h_tm1 + (1 - z) * hh  # Previous and candidate state mixed by update gate.
		
		return {'r': r, 'z': z, 'h': h, 'hh': hh}
	
	#text = seq2seq.encoder_in_test
	empty_container = np.empty([sample_index.shape[0], sample_index.shape[1], seq2seq.tgt_max_len, seq2seq.units])
	container = {'r': np.copy(empty_container), 
				'z': np.copy(empty_container),
				'h': np.copy(empty_container),
				'hh': np.copy(empty_container)}

	weight = seq2seq.decoder_model.get_layer('decoder_gru').get_weights()
	dec_layer_model = Model(inputs=seq2seq.decoder_model.input, 
							outputs=seq2seq.decoder_model.get_layer('decoder_gru').get_output_at(-1))
	emb_layer_model = Model(inputs=seq2seq.decoder_model.get_layer('decoder_emb').get_input_at(-1), 
							outputs=seq2seq.decoder_model.get_layer('decoder_emb').output)

	for i in range(sample_index.shape[0]):
		sample = seq2seq.encoder_in_test[sample_index[i]]
		decoder_states = seq2seq.encoder_model.predict(sample, batch_size=256)[0] 
		decoder_inputs = np.full([sample_index.shape[1], 1], SOS_token) # first token is SOS

		for t in range(seq2seq.tgt_max_len):
			# keras_h = dec_layer_model.predict([decoder_inputs] + [decoder_states], verbose = 0)
			# keras_output = keras_h[0].reshape([keras_h[0].shape[0], keras_h[0].shape[2]])

			decoder_emb = np.squeeze(emb_layer_model.predict(decoder_inputs, verbose=0, batch_size=256), axis=1)
			one_gate_values = propagate_gru(weight, decoder_emb, decoder_states, seq2seq.units)
			output_tokens, decoder_states = seq2seq.decoder_model.predict([decoder_inputs] + [decoder_states], batch_size=256, verbose=0)
			sampled_token_index = np.argmax(output_tokens[:, -1, :], axis=-1)  # Sample a token.
			decoder_inputs[:, 0] = sampled_token_index[:]
			for gate_name in ['r', 'z', 'h', 'hh']: 
				container[gate_name][i, :, t] = K.get_value(one_gate_values[gate_name])
	print("\t(Gate value).shape =", container['r'].shape)
	return container


"""
def get_encoder_state(seq2seq, samples, SOS_token=1):
	#Only get the last time step of encoder.
	first_key = list(samples.keys())[0]  
	container = np.zeros([len(samples), len(samples[first_key]), seq2seq.units])
	print("(Encoder) container.shape =", container.shape)

	# Store each class in for loop.
	for key_index, key in enumerate(sorted(samples.keys())):
		print("(Encoder), key =", key_index, key)
		encoder_output = seq2seq.encoder_model.predict(samples[key])[0] 
		container[key_index, :, :] = encoder_output
	print('container =', container.shape)
	return container
"""

