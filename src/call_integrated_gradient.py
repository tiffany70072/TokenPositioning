import keras.losses
import numpy as np 
import os
import pdb

#from IntegratedGradients import *
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import gradients
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras import backend as K

import gru_integrated_gradient as gruig


def linearly_interpolate(sample, reference=False, num_steps=50):
	# Use default reference values if reference is not specified
	if reference is False: 
		reference = np.zeros(sample.shape);

	# Reference and sample shape needs to match exactly
	assert sample.shape == reference.shape

	# Calcuated stepwise difference from reference to the actual sample.
	ret = np.zeros(tuple([num_steps] + [i for i in sample.shape]))
	for s in range(num_steps):
		ret[s] = reference + (sample - reference) * (s * 1.0 / num_steps)

	return ret, num_steps, (sample - reference) * (1.0 / num_steps)


def compute_ig_steps(decoder_model, decoder_states_container, decoder_inputs_container, 
					target_class, reference=False, k=64):
	"""Calculate the IG of output(output_t) to hidden_state(input_t).
	Arguments:
		decoder_states_container: Store the values of hidden state "with whole time steps". Starts from encoder output.
		decoder_inputs_container: Store the one-hot inputs with whole time steps. Should starts from SOS token.
		target_class: An integers.
	"""
	sess = tf.compat.v1.keras.backend.get_session()
	interpolate, num_steps, step_size = linearly_interpolate(decoder_states_container)  # (50, N, 10), int, (N, 10)
	#result = np.zeros(decoder_states_container.shape)
	
	print("\tClass index =", target_class)
	gradient = gradients(decoder_model.output[0][:, 0, target_class], decoder_model.input[1])
	result = np.zeros(decoder_states_container.shape)
	#print("result =", result.shape, decoder_states_container.shape)
	for i in range(num_steps):
		#print(decoder_inputs_container.shape, interpolate[i].shape)
		x = sess.run(gradient[0], feed_dict={decoder_model.input[0]: decoder_inputs_container, 
											decoder_model.input[1]: interpolate[i]})	
		result += x
	result = np.multiply(result, step_size)
	result = np.mean(result, axis=0)
	return result


# MOVE TO IG CODE.
def get_decoder_output(decoder_model, embedding, tgt_max_len, SOS_token=1, EOS_token=2):
	"""
	Return:
		decoder_inputs_container: Shape = (time_steps, batch, 1). Starts from SOS token.
		decoder_states_container: Shape = (time_steps, batch, units). Starts from encoder output.
	"""
	decoder_states = K.get_value(embedding)
	decoder_inputs = np.full([embedding.shape[0], 1], SOS_token) # first token is SOS
	decoder_inputs_container = np.empty([tgt_max_len+1, embedding.shape[0], 1])  # Shape = (Sample number, time_step)
	decoder_inputs_container[0] = decoder_inputs[:]
	decoder_states_container = np.empty([tgt_max_len+1, embedding.shape[0], embedding.shape[1]])
	decoder_states_container[0] = decoder_states

	for i in range(tgt_max_len):
		output_tokens, decoder_states = decoder_model.predict([decoder_inputs] + [decoder_states], verbose=0)
		sampled_token_index = np.argmax(output_tokens[:, -1, :], axis=-1)  # Sample a token.
		decoder_inputs[:, 0] = sampled_token_index[:]
		decoder_inputs_container[i+1, :, 0] = sampled_token_index
		decoder_states_container[i+1, :] = decoder_states
	decoder_inputs_transpose = np.transpose(decoder_inputs_container, [1, 0, 2])  # Change (time, batch, 1) to (batch, time, 1)
	decoder_inputs_transpose = np.squeeze(decoder_inputs_transpose, axis=-1)
	return decoder_states_container, decoder_inputs_transpose


def get_state_by_sample_index(seq2seq, sample_index):
    sample = seq2seq.encoder_in_test[sample_index]
    embedding = seq2seq.encoder_model(sample)[0]  # Shape=(100, 10), (100, 22, 10)
    decoder_states, decoder_inputs = get_decoder_output(seq2seq.decoder_model, embedding, tgt_max_len=seq2seq.tgt_max_len)
    return decoder_states, decoder_inputs


def get_model_without_argmax(seq2seq, input_t, output_t):
	"""output_t, input_t: t = 0 represents the time step of encoder output."""
	# decoder_model = [decoder_inputs] + [decoder_state_input_h], [decoder_outputs] + [state_h]	
	assert input_t <= output_t, "Time step incompatible. Cannot calculate integrated gradients."
	print("\n\tCross step. input_t = %d, output_t = %d" % (input_t, output_t))
	
	decoder_model = gruig.build_decoder_model_without_argmax(seq2seq, input_t, output_t)
	for layer_name in ["decoder_emb", "decoder_gru", "output_dense"]:
		try:
			weights = seq2seq.decoder_model.get_layer(layer_name).get_weights()
			decoder_model.get_layer(layer_name).set_weights(weights)
		except ValueError:
			pass
	return decoder_model  


def compute_multiple_target_class(result_list):
    total_result = np.zeros(result_list[0].shape)
    for result in result_list:
        total_result += result
    total_result /= float(len(result_list))
    return total_result


def get_important_neurons_by_IG(result, k=64):
    score = np.abs(result)
    #print(result.shape, score.shape)
    selected = list(np.argsort(score)[::-1][:k])
    #print("\tselected =", selected)
    #print(np.sort(score)[::-1][:4])	
    return selected


def compute_ig_within_GRU(seq2seq, decoder_states_container, decoder_inputs_container, real_h, target_gate="z",
						 target_class=[0], reference=False, k=32):
	"""
	Compute ig within GRU for several gates.
	Arguments:
		decoder_states_container, decoder_inputs_container: For only one time step, the inputs of decoder GRU cell.
		target_gate: A list of {"h", "r", "z"}.
		target_class: A list of int.
	"""
	assert target_gate == "z" or target_gate == "r" or target_gate == "h", "No this gate."
	print("\nCompute on %s gate." % target_gate)
	weight = seq2seq.decoder_model.get_layer("decoder_gru").get_weights()
	emb_layer_model = Model(inputs=seq2seq.decoder_model.get_layer('decoder_emb').get_input_at(-1), 
							outputs=seq2seq.decoder_model.get_layer('decoder_emb').output)
	inputs = np.squeeze(emb_layer_model.predict(decoder_inputs_container, verbose=0), axis=1)
	states = np.copy(decoder_states_container)
	h_tm1, h, z, r, hh, x_h, split_recurrent_h = gruig.get_GRU_components(inputs, states, weight)
	weight_array = np.concatenate([weight[0], weight[1], np.expand_dims(weight[2], axis=0)], axis=0)
	units = seq2seq.units
	if target_gate == "h":
		gate_model = gruig.build_GRU_with_h_gate_model(seq2seq)
		gate_model.get_layer("wx_h").set_weights([weight[0][:, units * 2:], weight[2][units * 2:]])
		gate_model.get_layer("uh_h").set_weights([weight[1][:, units * 2:]])
		y = gate_model.predict([h_tm1, inputs, z, r], steps=1)
		feed_dict = {gate_model.input[1]: inputs,
					gate_model.input[2]: z,
					gate_model.input[3]: r,}
		real = real_h
	elif target_gate == "z":
		gate_model = gruig.build_GRU_with_z_gate_model(seq2seq, weight_array)
		y = gate_model.predict([h_tm1, inputs, r, hh], steps=1)
		feed_dict = {gate_model.input[1]: inputs,
					gate_model.input[2]: r,
					gate_model.input[3]: hh}
		real = y
	elif target_gate == "r":
		gate_model = gruig.build_GRU_with_r_gate_model(seq2seq, weight_array)
		y = gate_model.predict([h_tm1, inputs, z, x_h, split_recurrent_h], steps=1)
		feed_dict = {gate_model.input[1]: inputs,
					gate_model.input[2]: z,
					gate_model.input[3]: x_h,
					gate_model.input[4]: split_recurrent_h}
		real = y
	assert np.mean(np.abs(y - real)) < 1e-6, "Wrong computation for error = %.8f" % np.mean(np.abs(y - real))
	print("delta =", np.mean(np.abs(y - real)))
	#assert np.mean(np.abs(y - h)) < 1e-6, "Wrong computation for error = %.8f" % np.mean(np.abs(y - h))
		
	interpolate, num_steps, step_size = linearly_interpolate(decoder_states_container)  # (50, N, 10), int, (N, 10)
	result = np.zeros(decoder_states_container.shape)  # (N, 256), 
	total_result = np.zeros(decoder_states_container.shape)
	sess = K.get_session()
	
	for class_ in target_class:
		print("Class index =", class_)
		gradient = gradients(gate_model.output[:, class_], gate_model.input[0])
		result = np.zeros(decoder_states_container.shape)
		for i in range(num_steps):
			feed_dict[gate_model.input[0]] = interpolate[i]
			x = sess.run(gradient[0], feed_dict=feed_dict)
			result += x
		result = np.multiply(result, step_size)

		
		"""if target_gate == "h":
			score = np.abs(np.mean(result, axis=0))
			print("selected =", list(np.argsort(score)[::-1][:k]))
			print("other =", list(np.argsort(score)[::-1][k:]))
			print(np.sort(score)[::-1][:4])	"""
		total_result += result
	total_result /= float(len(target_class))
	score = np.abs(np.mean(total_result, axis=0))
	print("(total) selected =", list(np.argsort(score)[::-1][:k]))
	#print("other =", list(np.argsort(score)[::-1][k:]))
	print(np.sort(score)[::-1][:4])	
	return score


"""
def compute_ig_HtoH(model, last_hidden, decoder_inputs, target_class, time_step=0, reference=False, k=32, verbose=1):
	#Compute IG for h(t)_feat with h(t-1).
	
	last_hidden = K.get_value(last_hidden)
	sess = tf.compat.v1.keras.backend.get_session()
	interpolate, num_steps, step_size = linearly_interpolate(last_hidden)  # (50, N, 10), int, (N, 10)

	for feat in target_class:
		print("Feat =", feat, end=": ")
		gradient = gradients(model.output[1][:, feat], model.input[1])
		result = np.zeros(last_hidden.shape)
		for i in range(num_steps):
			result += sess.run(gradient[0], feed_dict={model.input[0]: decoder_inputs, 
				model.input[1]: interpolate[i]})
			
		#result = np.multiply(result, step_size)
	result = np.multiply(result, step_size)
	result /= len(target_class)
	
	return  result
"""

"""
def test_function(seq2seq, selector):
	conditions = selector.get_word_condition(5)
	sample, sample_index = selector.get_sample(conditions)

	score_container = np.zeros([4, 256])
	for w in [3, 4, 5, 6]:
		src = sample[w]
		embedding = seq2seq.encoder_model(src)[0]
		decoder_states_container, decoder_inputs_container = get_decoder_output(seq2seq.decoder_model, embedding, tgt_max_len=seq2seq.tgt_max_len)	
		print(decoder_states_container[3][0, :10])

		decoder_inputs_transpose = np.transpose(decoder_inputs_container, [1, 0, 2])  # Change (time, batch, 1) to (batch, time, 1)
		decoder_inputs_transpose = np.squeeze(decoder_inputs_transpose, axis=-1)
		
		for t in [4]:
			print("\n\nt =", t)
			
			score = compute_ig_steps(seq2seq, t, 5,
									decoder_states_container, 
									decoder_inputs_transpose, 
									target_class=[3])

			
			score = compute_ig_HtoH(seq2seq.decoder_model, 
									decoder_states_container[3], 
									decoder_inputs_container[3], 
									target_class=position_1[:64])
			
			score = np.abs(score)
			argsort_ = np.argsort(score)[::-1]
			sort_ = np.sort(score)[::-1]
			
			score_container[w-3] = score
	np.save("score_w36.npy", score_container)
"""	
	
			




