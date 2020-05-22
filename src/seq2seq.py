""" The main part of the Seq2Seq model construction and training.
"""

#import glob  # Search all matched files under a directory.
import keras.losses  # TODO: Remove this repeated import.
import numpy as np
import os  # For makedir.
import pdb
#import sys
from pathlib import Path
from tensorflow.compat import v1 as tf
tf.disable_v2_behavior()


import data_loader
import build_model
import utils
#import word_utils


class Seq2Seq(object):
	
	def __init__(self, args):
		self.task = args.task 	# Default: "task1".
		self.units = args.units 	# Default: 128.
		self.model_path = args.model_path
		self.mode = args.mode
		self.data_name = args.data_name
		
		self.get_data()
		self.seq2seq_model, self.encoder_model, self.decoder_model = build_model.seq2seq(
			self.src_max_len, self.tgt_max_len, self.src_token_size, self.tgt_token_size, 
			latent_dim=self.units)
			
		
	def get_data(self):
		if self.task == "autoencoder":
			import data_preprocessing_autoencoder
			self = data_preprocessing_autoencoder.main(self)

		elif self.task == "autoenc-last" or self.task == 'token-posi' or self.task == "eos-posi" or self.task == "rhy-posi":
			print("get_data")
			self.encoder_in, self.decoder_in, self.decoder_out = data_loader.load_data(task=self.task, data_name=self.data_name, data_type="train")
			self.encoder_in_valid, self.decoder_in_valid, self.decoder_out_valid = data_loader.load_data(task=self.task, data_name=self.data_name, data_type="valid")
			self.encoder_in_test, self.decoder_in_test, self.decoder_out_test = data_loader.load_data(task=self.task, data_name=self.data_name, data_type="valid")
			
			assert self.encoder_in.shape[1] == self.encoder_in_valid.shape[1], "Data size not match"
			assert self.decoder_in.shape[1] == self.decoder_in_valid.shape[1], "Data size not match"
            
			self.src_max_len = self.encoder_in.shape[1]
			self.tgt_max_len = self.decoder_out.shape[1]
			token_size = int(max(np.max(self.encoder_in), np.max(self.decoder_in))) + 1  # Token size of autoencoder are same between encoder data and decoder data.         
			self.src_token_size = token_size
			self.tgt_token_size = token_size

		elif (task == "control_length" or task == "control_length_content" 
			or task == "control_rhyme_content" or task == "control_pos_content" or task == "toy_pos_signal"):
			self.src_token_size = np.max(self.encoder_in) + 1  # TODO: Remove this if/else.
			self.tgt_token_size = np.max(self.decoder_out) + 1
		print("(Load data) token_size  =", self.src_token_size, self.tgt_token_size)
	
	def load_seq2seq(self, model_path, epoch=None):
		keras.losses.custom_loss = utils.masked_perplexity_loss
		if epoch is None:
			self.seq2seq_model.load_weights(os.path.join(model_path, "seq2seq.h5")) 
		else:
			self.seq2seq_model.load_weights(os.path.join(model_path, "epoch" + str(epoch) + "_seq2seq.h5")) 
		

	def save_seq2seq(self, epoch=None):
		"""Save trained Seq2Seq model."""

		# Check whether path exists.
		file_path = Path(self.model_path)
		if not file_path.exists():
			os.mkdir(self.model_path)
		if epoch is None:
			self.seq2seq_model.save_weights(os.path.join(self.model_path, "seq2seq.h5")) 
			#self.encoder_model.save_weights(os.path.join(self.model_path, "encoder.h5"))
			#self.decoder_model.save_weights(os.path.join(self.model_path, "decoder.h5"))
		else:
			self.seq2seq_model.save_weights(os.path.join(self.model_path, "epoch" + str(epoch) + "_seq2seq.h5")) 
			#self.encoder_model.save_weights(os.path.join(self.model_path, "epoch" + str(epoch) + "_encoder.h5"))
			#self.decoder_model.save_weights(os.path.join(self.model_path, "epoch" + str(epoch) + "_decoder.h5"))


	"""
	def quick_validation(N=100):
		pred = self.seq2seq_model.predict([self.encoder_in_test[:N], self.decoder_in_test[:N], np.zeros((N, 1))])
		utils.compute_accuracy(self.decoder_out_test[:N], pred)
		# TODO: check this function.
	"""
	

	def inference_batch(self, input_seq, SOS_token=1, EOS_token=2, output=False):
		states_value = self.encoder_model.predict(input_seq, batch_size=256)[:-1]  # Encode the input as state vectors. A list.
		
		target_seq = np.zeros((input_seq.shape[0], 1)) # Generate empty target sequence of length 1.
		target_seq[:, 0] = np.tile(SOS_token, (input_seq.shape[0]))

		# Sampling loop for a batch of sequences (to simplify, here we assume a batch of size 1).
		stop_condition = np.zeros([input_seq.shape[0]])
		decoded_sentence = []
		for i in range(target_seq.shape[0]):
			decoded_sentence += [[]]

		while np.sum(stop_condition) < input_seq.shape[0]:
			decoder_outputs = self.decoder_model.predict([target_seq] + states_value, verbose=0, batch_size=256)
			
			if len(decoder_outputs) == 2:
				output_tokens = decoder_outputs[0]
				states_value = [decoder_outputs[1]]
			elif len(decoder_outputs) == 3:
				output_tokens = decoder_outputs[0]
				states_value = decoder_outputs[1:]
			
			sampled_token_index = np.argmax(output_tokens[:, -1, :], axis=-1) # Sample a token
			
			for i in range(target_seq.shape[0]):
				if (sampled_token_index[i] == EOS_token or len(decoded_sentence[i]) > self.tgt_max_len):
					decoded_sentence[i] += [sampled_token_index[i]]
					stop_condition[i] = 1
				elif stop_condition[i] == 0: 
					decoded_sentence[i] += [sampled_token_index[i]]  # (+= sampled_char)
					target_seq[i, 0] = sampled_token_index[i]
			if int(np.sum(stop_condition)) == input_seq.shape[0]: 
				break

		return decoded_sentence

	