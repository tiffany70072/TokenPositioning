import numpy as np 
import pdb

from tensorflow.keras.activations import hard_sigmoid
#from tensorflow.keras.activations import tanh
from tensorflow.compat.v1.keras.activations import tanh
from tensorflow.compat.v1.keras.layers import Dense, Embedding, Input, Lambda
from tensorflow.compat.v1.keras.layers import GRU 
from tensorflow.compat.v1.keras import layers
#from tensorflow.compat.v1.keras.engine.topology import Layer
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras import backend as K


def slice(x, index):
	return x[:, index:index+1]


def build_decoder_model_without_argmax(seq2seq, input_t, output_t):
	# Remove all initializer.
	input_state = Input(shape=(seq2seq.units,), name="decoder_state") 
	decoder_inputs = Input(shape=(None,), name="decoder_input") 
	decoder_embedding = Embedding(seq2seq.tgt_token_size, seq2seq.units, input_length=None, name="decoder_emb")
	decoder_gru = GRU(seq2seq.units, return_sequences=True, return_state=True, name="decoder_gru")
	decoder_dense = Dense(seq2seq.tgt_token_size, activation="softmax", name="output_dense")

	state = input_state
	for t in range(input_t, output_t):
		inputs = Lambda(slice, arguments={"index": t})(decoder_inputs)  # Count encoder output as time 0.
		inputs_embedding = decoder_embedding(inputs)
		decoder_outputs_time, state = decoder_gru(inputs_embedding, initial_state=state)
	if input_t == output_t:
		decoder_outputs_time = Lambda(lambda x: K.expand_dims(x, axis=1))(state)
	softmax = decoder_dense(decoder_outputs_time)
	decoder_model = Model([decoder_inputs, input_state], [softmax] + [state])
	
	return decoder_model


def get_GRU_components(inputs, states, weight):
	units = weight[0].shape[0]

	kernel = K.variable(weight[0])  # shape = (input_dim, self.units * 3)
	recurrent_kernel = K.variable(weight[1])  # shape = (self.units, self.units * 3)
	bias = K.variable(weight[2])  # bias_shape = (3 * self.units,)
	inputs = K.variable(inputs)	 # Not sure.
	h_tm1 = K.variable(states)	  # Previous memory state.
		
	# Update gate.
	kernel_z = kernel[:, :units]
	recurrent_kernel_z = recurrent_kernel[:, :units]
	input_bias_z = bias[:units]
	# Reset gate.
	kernel_r = kernel[:, units:units * 2]
	recurrent_kernel_r = recurrent_kernel[:, units:units * 2]
	input_bias_r = bias[units: units * 2]
	# New gate.
	kernel_h = kernel[:, units * 2:]
	recurrent_kernel_h = recurrent_kernel[:, units * 2:]
	input_bias_h = bias[units * 2:]

	x_z = K.bias_add(K.dot(inputs, kernel_z), input_bias_z)
	x_r = K.bias_add(K.dot(inputs, kernel_r), input_bias_r)
	x_h = K.bias_add(K.dot(inputs, kernel_h), input_bias_h)	   
	recurrent_z = K.dot(h_tm1, recurrent_kernel_z)
	recurrent_r = K.dot(h_tm1, recurrent_kernel_r)
					
	z = hard_sigmoid(x_z + recurrent_z)  # Recurrent activation = 'hard_sigmoid'.
	r = hard_sigmoid(x_r + recurrent_r)

	recurrent_h = K.dot(r * h_tm1, recurrent_kernel_h)
	# Get split part of recurrent_h.
	split_recurrent_h = K.expand_dims(h_tm1, axis=-1) * recurrent_kernel_h
	r_unsqueeze = K.expand_dims(r, axis=-1)
	recompute_recurrent_h = K.sum(r_unsqueeze * split_recurrent_h, axis=1)
	#print(recurrent_h.shape, h_tm1.shape, recurrent_kernel_h.shape, split_recurrent_h.shape)
	#print(K.get_value(recompute_recurrent_h)[0, :3], np.mean(K.get_value(recompute_recurrent_h)))
	#print(K.get_value(recurrent_h)[0, :3], np.mean(K.get_value(recurrent_h)))
	delta = np.mean(np.abs(K.get_value(recompute_recurrent_h) - K.get_value(recurrent_h)))
	print("delta =", delta, np.mean(K.get_value(recompute_recurrent_h)), np.mean(K.get_value(recurrent_h)))
	assert delta < 1e-6, "r gate is wrong."
	
	hh = tanh(x_h + recurrent_h) 	# Activation = 'tanh'.	   
	# Previous and candidate state mixed by update gate.
	h = z * h_tm1 + (1 - z) * hh

	return K.get_value(h_tm1), K.get_value(h), K.get_value(z), K.get_value(r), K.get_value(hh), K.get_value(x_h), K.get_value(split_recurrent_h)

"""
class GRU_with_h_gate(Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        super(GRU_with_h_gate, self).__init__(**kwargs)

    def build(self, input_shape):
    	# Useless to set initializer and trainable parameters.
    	self.kernel_h = self.add_weight(name="kernel", shape=(input_shape[-1], self.units))
    	self.recurrent_h = self.add_weight(name="recurrent_kernel", shape=(input_shape[-1], self.units))
        self.input_bias_h = self.add_weight(name="bias", shape=(self.units,))
        #weights=[embedding_matrix],

        super(GRU_with_h_gate, self).build(input_shape)  # Must call this function in the end.

    def call(self, z, r, h_tm1):
    	#z_input = Input(shape=(seq2seq.units,), name="z_input") 
		#r_input = Input(shape=(seq2seq.units,), name="r_input") 
		#h_tm1_input = Input(shape=(seq2seq.units,), name="h_input") 
		#x_h = layers.Add()([layers., ])
		x_h = K.bias_add(K.dot(h_tm1, self.kernel_h), self.input_bias_h)	  
		recurrent_h = K.dot(r * h_tm1, recurrent_kernel_h)
		hh = tanh(x_h + recurrent_h) 	# Activation = 'tanh'.	   
		h = z * h_tm1 + (1 - z) * hh
        return h

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
"""
"""
def build_GRU_with_h_gate_model(seq2seq, weight_array):
	h_tm1_input = Input(shape=(seq2seq.units,), name="h_input")
	x_input = Input(shape=(seq2seq.units,), name="x_input") 
	z_input = Input(shape=(seq2seq.units,), name="z_input") 
	r_input = Input(shape=(seq2seq.units,), name="r_input") 
	
	def gru_with_h_gate(x, weight):
		h_tm1 , inputs, z, r = x[0], x[1], x[2], x[3]
		weight = K.variable(weight)
		units = h_tm1.shape[-1]
		
		kernel_h = weight[:units, units * 2:]
		recurrent_kernel_h = weight[units: units * 2, units * 2:]
		input_bias_h = weight[units * 2, units * 2:]  # Change to 1 dim.
		x_h = K.bias_add(K.dot(inputs, kernel_h), input_bias_h)	  
		recurrent_h = K.dot(r * h_tm1, recurrent_kernel_h)
		hh = tanh(x_h + recurrent_h) 	# Activation = 'tanh'.	   
		h = z * h_tm1 + (1 - z) * hh
		return h #, z, r, hh, x_h, kernel_h, recurrent_kernel_h, input_bias_h
		
	h = Lambda(gru_with_h_gate, arguments={"weight": weight_array})([h_tm1_input, x_input, z_input, r_input])
	GRU_with_h_gate_model = Model([h_tm1_input, x_input, z_input, r_input], h)
	print("h gate model.")
	GRU_with_h_gate_model.summary()
	return GRU_with_h_gate_model
"""

def build_GRU_with_h_gate_model(seq2seq):  # A new one.
	units = seq2seq.units
	h_tm1_input = Input(shape=(units,), name="h_input")
	x_input = Input(shape=(units,), name="x_input") 
	z_input = Input(shape=(units,), name="z_input") 
	r_input = Input(shape=(units,), name="r_input") 
	
	x_h = Dense(units, name="wx_h")(x_input)  # x_h = K.bias_add(K.dot(inputs, kernel_h), input_bias_h)	 
	r_h_tm1 = layers.Multiply()([r_input, h_tm1_input])  # r * h_tm1
	recurrent_h = Dense(units, use_bias=False, name="uh_h")(r_h_tm1)  # recurrent_h = K.dot(r * h_tm1, recurrent_kernel_h)
	hh_ = layers.Add()([x_h, recurrent_h])
	hh = tanh(hh_) 	# hh = tanh(x_h + recurrent_h)
	h1 = layers.Multiply()([z_input, h_tm1_input])
	h2 = layers.Multiply()([1 - z_input, hh])
	h = layers.Add()([h1, h2])  # h = z * h_tm1 + (1 - z) * hh
	GRU_with_h_gate_model = Model([h_tm1_input, x_input, z_input, r_input], h)
	#print("h gate model.")
	#GRU_with_h_gate_model.summary()
	return GRU_with_h_gate_model


def build_GRU_with_z_gate_model(seq2seq, weight_array):
	h_tm1_input = Input(shape=(seq2seq.units,), name="h_input") 
	x_input = Input(shape=(seq2seq.units,), name="x_input") 
	r_input = Input(shape=(seq2seq.units,), name="r_input") 
	hh_input = Input(shape=(seq2seq.units,), name="hh_input") 

	def gru_with_z_gate(x, weight):
		h_tm1, inputs, r, hh = x[0], x[1], x[2], x[3]
		weight = K.variable(weight)
		units = h_tm1.shape[-1]
		
		kernel_z = weight[:units, :units]
		recurrent_kernel_z = weight[units:units * 2, :units]
		input_bias_z = weight[units * 2, :units]  # Change to 1 dim.	   
		x_z = K.bias_add(K.dot(inputs, kernel_z), input_bias_z)	  
		recurrent_z = K.dot(h_tm1, recurrent_kernel_z)
		z_without_activate = x_z + recurrent_z
		z = hard_sigmoid(z_without_activate)
		h = z * h_tm1 + (1 - z) * hh
		#return h
		return z
	output = Lambda(gru_with_z_gate, arguments={"weight": weight_array})([h_tm1_input, x_input, r_input, hh_input])
	#h = layers.Add()([h_tm1_input, x_input])
	gate_model = Model([h_tm1_input, x_input, r_input, hh_input], output)
	#print("z gate model.")
	#gate_model.summary()
	return gate_model


def build_GRU_with_r_gate_model(seq2seq, weight_array):
	h_tm1_input = Input(shape=(seq2seq.units,), name="h_input") 
	x_input = Input(shape=(seq2seq.units,), name="x_input") 
	z_input = Input(shape=(seq2seq.units,), name="z_input") 
	xh_input = Input(shape=(seq2seq.units,), name="xh_input")  # x_h = K.bias_add(K.dot(inputs, kernel_h), input_bias_h)
	rh_input = Input(shape=(seq2seq.units, seq2seq.units), name="rh_input")  # split_recurrent_h = K.dot(h_tm1.transpose(), recurrent_kernel_h)
	
	def gru_with_r_gate(x, weight):
		h_tm1, inputs, z, x_h, split_recurrent_h = x[0], x[1], x[2], x[3], x[4]
		weight = K.variable(weight)
		units = h_tm1.shape[-1]
		
		kernel_r = weight[:units, units:units * 2]
		recurrent_kernel_r = weight[units:units * 2, units:units * 2]
		input_bias_r = weight[units * 2, units:units * 2]  # Change to 1 dim.	   
		x_r = K.bias_add(K.dot(inputs, kernel_r), input_bias_r)	  
		recurrent_r = K.dot(h_tm1, recurrent_kernel_r)
		r_without_activate = x_r + recurrent_r
		r = hard_sigmoid(r_without_activate)
		#r = hard_sigmoid(x_r + recurrent_r)

		# Recompute recurrent_h by two parts.
		r_unsqueeze = K.expand_dims(r, axis=-1)
		recompute_recurrent_h = K.sum(r_unsqueeze * split_recurrent_h, axis=1)
		hh = tanh(x_h + recompute_recurrent_h)

		h = z * h_tm1 + (1 - z) * hh
		#return h
		return r

	output = Lambda(gru_with_r_gate, arguments={"weight": weight_array})([h_tm1_input, x_input, z_input, xh_input, rh_input])
	gate_model = Model([h_tm1_input, x_input, z_input, xh_input, rh_input], output)
	#print("r gate model")
	#gate_model.summary()
	return gate_model
