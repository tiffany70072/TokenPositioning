#import collections
#import matplotlib
#matplotlib.use('Agg')
#from  matplotlib import pyplot as plt
import numpy as np 
#import pdb
#from sklearn.decomposition import PCA

#import analysis
#import select_neurons
#import utils
#import visualization

from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K


def dense_forward(kernel, bias, h):
	h = K.variable(h)
	output = K.dot(h, kernel)    
	output = K.bias_add(output, bias)
	output = K.get_value(output)
	output = np.argmax(output, axis = 1)
	return output


def verify_decoder(seq2seq, sample, selected_feature, time_step, mode="disable", replace_by="zero", SOS_token=1, verbose=1, given_state=None):
	# TDOO: other modes + replace other gates + replace then keep moving forward.

	
	if mode == "enable":
		if verbose == 1:
			print("\tEnable:", selected_feature[:5], len(selected_feature), "at t = %d" % time_step, end=", ")
			print("\tOther replaced by", replace_by)
		elif verbose == 2:
			print("enable", end="\t")
		selected_feature = list(set([i for i in range(seq2seq.units)]) - set(selected_feature))
	else:
		if verbose == 1:        
			print("\tDisable:", selected_feature[:5], len(selected_feature), "at t = %d" % time_step, end=", ")
			print("\tReplaced by", replace_by)
		elif verbose == 2:
			print("disable", end="\t")

	output_container = np.ones([len(sample), seq2seq.tgt_max_len])  # For return.

	# Before "time_step": 0 -> 1 and 1 -> 2.
	decoder_states = seq2seq.encoder_model.predict(sample, batch_size=256)[0]  # decoder_states[0] = encoder.output	
	decoder_inputs = np.full([len(sample), 1], SOS_token) # first token is SOS
	for t in range(time_step-1):  # Run normal decoder for "time_step" - 1 times.
		output_tokens, decoder_states = seq2seq.decoder_model.predict([decoder_inputs] + [decoder_states], verbose=0)
		sampled_token_index = np.argmax(output_tokens[:, -1, :], axis=-1) # Sample a token
		decoder_inputs[:, 0] = sampled_token_index[:]
		output_container[:, t] = sampled_token_index[:]

	# Replace neurons: 2 -> 3 (before FC)
	dec_layer_model = Model(inputs=seq2seq.decoder_model.input, 
							outputs=seq2seq.decoder_model.get_layer('decoder_gru').get_output_at(-1))
	hidden = dec_layer_model.predict([decoder_inputs] + [decoder_states], verbose=0, batch_size=256)[1]
	# dec_layer_model.output.shape = hidden[0].shape, hidden[1].shape)  # (N, 1, dim), (N, dim)
	if replace_by == "zero":
		hidden[:, selected_feature] *= 0
	elif replace_by == "last_h":  # TODO: check correctness
		hidden[:, selected_feature] = decoder_states[:, selected_feature]
	elif replace_by == "negative":
		hidden[:, selected_feature] = -hidden[:, selected_feature]
	elif replace_by == "given":
		hidden[:, selected_feature] = given_state
		if verbose: print("given =", given_state.shape, hidden[:, selected_feature].shape)
	else:
		print("No this mode!!!")
		return
    
	weight = seq2seq.decoder_model.get_layer("output_dense").get_weights()
	output = dense_forward(kernel=K.variable(weight[0]), bias=K.variable(weight[1]), h=hidden)
	output_container[:, time_step-1] = output

	# After "time_step": 3 -> 4, ...
	decoder_inputs = np.expand_dims(output, axis=1)
	decoder_states = np.copy(hidden)
	for t in range(time_step, seq2seq.tgt_max_len):
		output_tokens, decoder_states = seq2seq.decoder_model.predict([decoder_inputs] + [decoder_states], verbose=0, steps=1, batch_size=256)
		sampled_token_index = np.argmax(output_tokens[:, -1, :], axis=-1) # Sample a token
		decoder_inputs[:, 0] = np.copy(sampled_token_index)
		output_container[:, t] = np.copy(sampled_token_index)

	return output_container


"""
def verify_trigger(seq2seq, sample, selected_feature, t1, t2, SOS_token=1):
	#Replace t2 feature to t1.
	# TDOO: other modes + replace other gates + replace then keep moving forward
	print("Trigger:", selected_feature[:5], len(selected_feature), "at t = %d" % t1)
	output_container = np.ones([len(sample), seq2seq.tgt_max_len])  # For return.

	# Before "time_step"
	decoder_states = seq2seq.encoder_model.predict(sample)[0]  # decoder_states[0] = encoder.output	
	decoder_inputs = np.full([len(sample), 1], SOS_token) # first token is SOS
	for t in range(t1-1):  # Run normal decoder for "time_step" - 1 times.
		output_tokens, decoder_states = seq2seq.decoder_model.predict([decoder_inputs] + [decoder_states], verbose=0)
		sampled_token_index = np.argmax(output_tokens[:, -1, :], axis=-1) # Sample a token
		decoder_inputs[:, 0] = sampled_token_index[:]
		output_container[:, t] = sampled_token_index[:]
	hidden_t1 = np.copy(decoder_states)
	decoder_inputs_t1 = np.copy(decoder_inputs)

	for t in range(t1-1, t2-1):
		output_tokens, decoder_states = seq2seq.decoder_model.predict([decoder_inputs] + [decoder_states], verbose=0)
		sampled_token_index = np.argmax(output_tokens[:, -1, :], axis=-1) # Sample a token
		decoder_inputs[:, 0] = sampled_token_index[:]

	hidden_t1[:, selected_feature] = decoder_states[:, selected_feature]
	# Replace neurons
	dec_layer_model = Model(inputs=seq2seq.decoder_model.input, 
							outputs=seq2seq.decoder_model.get_layer('decoder_gru').get_output_at(-1))
	hidden = dec_layer_model.predict([decoder_inputs_t1] + [hidden_t1], verbose=0)[1]
	weight = seq2seq.decoder_model.get_layer("output_dense").get_weights()
	output = dense_forward(kernel=K.variable(weight[0]), bias=K.variable(weight[1]), h=hidden)
	output_container[:, t1-1] = output

	return output_container


def verify_xh(seq2seq, SOS_token=1):
	import word_utils
	acc1 = []
	acc2 = []
	acc3 = []
	n = 200
	for bs in range(50):
		print(bs)

		idx1 = bs * n 
		idx2 = (bs + 1) * n
		sample = seq2seq.encoder_in_test[idx1:idx2]
		original_output = seq2seq.inference_batch(sample)
		output_container = np.ones([len(sample), seq2seq.tgt_max_len])  # For return.

		# Before "time_step"
		decoder_states = seq2seq.encoder_model.predict(sample)[0]  # decoder_states[0] = encoder.output	
		for t in range(seq2seq.tgt_max_len - 1):  # Run normal decoder for "time_step" - 1 times.
			#print(t)
			if t == 0:
				decoder_inputs = np.full([len(sample), 1], SOS_token) # first token is SOS
			else:
				for i in range(len(original_output)):
					#pdb.set_trace()
					if t < len(original_output[(i+1) % n]):
						decoder_inputs[i, 0] = original_output[(i+1) % n][t] 
			output_tokens, decoder_states = seq2seq.decoder_model.predict([decoder_inputs] + [decoder_states], verbose=0)
			sampled_token_index = np.argmax(output_tokens[:, -1, :], axis=-1) # Sample a token
			output_container[:, t] = sampled_token_index[:]
		
		sample_next_x = np.concatenate([seq2seq.encoder_in_test[idx1+1:idx2+1, :-2], sample[:, -2:]], axis=1)
		next_x_output = seq2seq.inference_batch(sample_next_x)
		#print(sample_next_x[:3])
		#print(sample[:3])
		#return
		
		acc1.append(word_utils.evaluate(seq2seq, real=sample, pred=original_output))
		acc2.append(word_utils.evaluate(seq2seq, real=sample, pred=output_container))
		acc3.append(word_utils.evaluate(seq2seq, real=sample, pred=next_x_output))
		
	acc1 = np.mean(np.array(acc1))
	acc2 = np.mean(np.array(acc2))
	acc3 = np.mean(np.array(acc3))
	print(acc1, acc2, acc3)
			
			
	

def verify_dense(seq2seq, common_rhyme, word_to_rhyme, rhyme_to_word, k=18):
	weight, bias = seq2seq.seq2seq_model.get_layer("output_dense").get_weights()
	# Get fixed length of word size of each rhyme token.
	for key in rhyme_to_word:
		word_size = len(rhyme_to_word[key])
		break

	null_container = np.empty([word_size, k])
	important_dense = collections.defaultdict(lambda: null_container)
	scores_dense = collections.defaultdict(lambda: np.zeros([128]))
	counter = collections.defaultdict(lambda: 0)
	
	for rhyme in common_rhyme:
		rhyme_index = seq2seq.src_ctoi[rhyme]
		for word_token in rhyme_to_word[rhyme_index]:
			largest_weight = np.argsort(np.abs(weight[:, word_token]))[::-1][:k]
			important_dense[rhyme_index][counter[rhyme_index]] = largest_weight
			scores_dense[rhyme_index] += weight[:, word_token]
			counter[rhyme_index] += 1
			#print("rhyme: %s, word: %d," % (rhyme, word_token), largest_weight)
		print(rhyme, rhyme_index, ", scores =", np.argsort(np.abs(scores_dense[rhyme_index]))[::-1][:k])
		print(np.sort(np.abs(scores_dense[rhyme_index]))[::-1][:k])
	#verify_dense_clustering(important_dense, seq2seq, common_rhyme, word_to_rhyme, rhyme_to_word)
	assert counter[key] == word_size, "Some count error in verfy dense."

		
	#for word_token in range(6, weight.shape[1]):  # tgt_token_size
	#	rhyme_index = word_to_rhyme[word_token]
	#	if rhyme_index != -1 and seq2seq.src_itoc[rhyme_index] in common_rhyme:
			
	pdb.set_trace()
		

	return

def verify_z_gate(seq2seq, selector, k=64):
	units = seq2seq.units
	weight = seq2seq.decoder_model.get_layer("decoder_gru").get_weights()
	weight = weight[1][:, :units]

	conditions = selector.get_word_condition(word_index=3)
	sample, sample_index = selector.get_sample(conditions)
	hidden_state = analysis.get_hidden_state(seq2seq, sample)

	#pdb.set_trace()
	IG_feature = [254, 220, 150, 197, 41, 20, 191, 93, 189, 34, 7, 108, 167, 178, 43, 180, 204, 182, 135, 146, 231, 213, 144, 157, 230, 209, 8, 217, 170, 195, 65, 101, 219, 74, 246, 179, 13, 136, 185, 223, 165, 175, 233, 66, 241, 244, 227, 232, 4, 70, 211, 39, 84, 161, 131, 229, 46, 115, 2, 253, 240, 75, 51, 134]
	RFE_feature = [6, 7, 89, 119, 158, 204, 230, 254, 14, 37, 41, 71, 182, 207, 208, 235, 29, 33, 136, 168, 169, 192, 198, 219, 8, 51, 57, 63, 105, 150, 220, 237, 5, 16, 43, 88, 90, 115, 130, 134, 66, 107, 118, 122, 171, 175, 178, 217, 13, 26, 35, 39, 65, 114, 157, 159, 58, 104, 112, 188, 202, 205, 234, 248]
	union = [5, 6, 7, 8, 13, 14, 16, 20, 29, 33, 34, 37, 41, 43, 51, 57, 63, 65, 71, 74, 88, 89, 90, 93, 101, 105, 108, 115, 119, 130, 134, 135, 136, 144, 146, 150, 157, 158, 167, 168, 169, 170, 178, 179, 180, 182, 185, 189, 191, 192, 195, 197, 198, 204, 207, 208, 209, 213, 217, 219, 220, 223, 230, 231, 235, 237, 246, 254]
	position_1 = [13, 106, 131, 136, 167, 180, 214, 219, 34, 47, 68, 78, 171, 178, 232, 235, 15, 115, 135, 182, 189, 192, 213, 233, 2, 9, 33, 39, 104, 114, 163, 216]
	cd = [ 39,  41, 148, 157, 178, 189, 191, 218, 2,  13,  27,  58,  65, 170, 175, 251, 22,  73,  87,  90, 125, 135, 205, 232, 34, 138, 167, 181, 192, 196, 200, 243]
	

	for t in [1, 2, 3, 4]:
		score = []
		hidden = hidden_state[3, t]  # length = 2, 3, 4, "5"
		for feature in [IG_feature, RFE_feature, union]:
			for i in feature:
				result = np.dot(hidden[:, position_1], weight[position_1, i])
				score.append(np.mean(result, axis=0))
			#print(score)
			print(t, np.mean(np.array(score)))
			#print(np.argsort(score)[::-1][:k])  # find the larger
			#print(np.argsort(score)[:k])   # find the smaller

def verify_r_gate(seq2seq, selector, k=64):
	units = seq2seq.units
	weight = seq2seq.decoder_model.get_layer("decoder_gru").get_weights()
	weight = weight[1][:, units:units * 2]  # r gate
	#weight = weight[1][:, units * 2:]  # h gate


	#conditions = selector.get_word_condition(word_index=3)
	selector.common_word = [3, 4, 5, 6, 7, 8, 9]
	conditions = selector.get_word_condition(length=5)
	sample, sample_index = selector.get_sample(conditions)
	hidden_state = analysis.get_hidden_state(seq2seq, sample)

	pdb.set_trace()
	import tmp
	weight = seq2seq.decoder_model.get_layer("decoder_gru").get_weights()
	dense_weight = seq2seq.decoder_model.get_layer("output_dense").get_weights()
	tmp.position_1(hidden, selector, weight, dense_weight)

	IG_feature = [254, 220, 150, 197, 41, 20, 191, 93, 189, 34, 7, 108, 167, 178, 43, 180, 204, 182, 135, 146, 231, 213, 144, 157, 230, 209, 8, 217, 170, 195, 65, 101, 219, 74, 246, 179, 13, 136, 185, 223, 165, 175, 233, 66, 241, 244, 227, 232, 4, 70, 211, 39, 84, 161, 131, 229, 46, 115, 2, 253, 240, 75, 51, 134]
	RFE_feature = [6, 7, 89, 119, 158, 204, 230, 254, 14, 37, 41, 71, 182, 207, 208, 235, 29, 33, 136, 168, 169, 192, 198, 219, 8, 51, 57, 63, 105, 150, 220, 237, 5, 16, 43, 88, 90, 115, 130, 134, 66, 107, 118, 122, 171, 175, 178, 217, 13, 26, 35, 39, 65, 114, 157, 159, 58, 104, 112, 188, 202, 205, 234, 248]
	union = [5, 6, 7, 8, 13, 14, 16, 20, 29, 33, 34, 37, 41, 43, 51, 57, 63, 65, 71, 74, 88, 89, 90, 93, 101, 105, 108, 115, 119, 130, 134, 135, 136, 144, 146, 150, 157, 158, 167, 168, 169, 170, 178, 179, 180, 182, 185, 189, 191, 192, 195, 197, 198, 204, 207, 208, 209, 213, 217, 219, 220, 223, 230, 231, 235, 237, 246, 254]
	position_1 = [13, 106, 131, 136, 167, 180, 214, 219, 34, 47, 68, 78, 171, 178, 232, 235, 15, 115, 135, 182, 189, 192, 213, 233, 2, 9, 33, 39, 104, 114, 163, 216]
	cd = [ 39,  41, 148, 157, 178, 189, 191, 218, 2,  13,  27,  58,  65, 170, 175, 251, 22,  73,  87,  90, 125, 135, 205, 232, 34, 138, 167, 181, 192, 196, 200, 243]
	fc = [212,  49, 209, 108, 227, 217,  70, 254, 215, 114, 204, 199, 171,
       235, 233,  91, 246, 221, 252, 191, 163, 229, 127,  30, 143, 151,
        87, 150, 152, 185, 166, 253, 205,  20, 231, 154,  29,  46, 138,
       148, 142,  43, 240, 244, 101,  16, 144,   3,  78,  73, 239,  74,
       126, 125, 122, 174, 195,  75,   6,  39, 223,  33, 164, 230]
	
	
	for t in [3, 4]:
		score = []
		hidden = hidden_state[3, t]  # length = 2, 3, 4, "5"
		for feature in [position_1]:
			for i in RFE_feature:
				#print(i)
				result = np.dot(hidden[:, feature], weight[feature, i])
				score.append(np.mean(result, axis=0))
			print(score[:5])
			print(t, np.mean(np.array(score)))
			#print(np.argsort(score)[::-1][:k])  # find the larger
			#print(np.argsort(score)[:k])   # find the smaller


def verify_dense_clustering(seq2seq, common_rhyme, word_to_rhyme, rhyme_to_word, k=20):
	
	import matplotlib
	import numpy as np 
	import os
	import pdb
	#from sklearn.decomposition import PCA
	#matplotlib.use('Agg')
	#import matplotlib.pyplot as plt


	weight, bias = seq2seq.seq2seq_model.get_layer("output_dense").get_weights()
	# Get fixed length of word size of each rhyme token.
	for key in rhyme_to_word:
		word_size = len(rhyme_to_word[key])
		break

	#null_container = np.empty([word_size, 128])
	#important_dense = collections.defaultdict(lambda: null_container)
	important_dense = np.empty([len(common_rhyme), word_size, 128])
	total_counter = 0
	counter = collections.defaultdict(lambda: 0)
	
	for rhyme in common_rhyme:
		rhyme_index = seq2seq.src_ctoi[rhyme]
		for word_token in rhyme_to_word[rhyme_index]:
			largest_weight = np.argsort(np.abs(weight[:, word_token]))[::-1][:k]
			tmp = np.zeros([128])
			for e in largest_weight:
				tmp[e] = 1
			#largest_weight = to_categorical(largest_weight, num_classes=128)
			#important_dense[total_counter][counter[rhyme_index]] = np.abs(weight[:, word_token])
			important_dense[total_counter][counter[rhyme_index]] = tmp
			#important_dense[rhyme_index][counter[rhyme_index]] = largest_weight
			counter[rhyme_index] += 1
			#print("rhyme: %s, word: %d," % (rhyme, word_token), largest_weight)
		

		x = important_dense[total_counter]
		#pdb.set_trace()
		print("x =", x.shape, x[:10])
		embedding = PCA(n_components=2).fit_transform(x)
		plt.scatter(embedding[:, 0], embedding[:, 1], s=10)
		plt.tight_layout()
		#print("path =", os.path.join(figure_path, "Figure_scatter_%s" % name))
		#plt.savefig(os.path.join(figure_path, "Figure_scatter_%s" % name))  # TODO: Chnage figure saved path.
		plt.show()
		plt.close()
		total_counter += 1

	x = important_dense
	x = x.reshape([-1, x.shape[-1]])
	embedding = PCA(n_components=2).fit_transform(x)
	embedding = embedding.reshape([-1, word_size, 2]) 
	for i in range(embedding.shape[0]):
		plt.scatter(embedding[i, :, 0], embedding[i, :, 1], s=10)
	plt.tight_layout()
	plt.show()

		
	#for word_token in range(6, weight.shape[1]):  # tgt_token_size
	#	rhyme_index = word_to_rhyme[word_token]
	#	if rhyme_index != -1 and seq2seq.src_itoc[rhyme_index] in common_rhyme:
			
	pdb.set_trace()
	
	return


def verify_rhyS(seq2seq, selector, feature_rhyS):
	length = 6
	conditions = selector.get_rhyme_condition(str(length))
	print("conditions =", conditions)
	sample, sample_index = selector.get_sample(conditions)
	hidden_state = analysis.get_hidden_state(seq2seq, sample)

	
	for feat in feature_rhyS:
		print("feat %d" % feat, end=", ")
		for t in range(1, 10):  # TODO: Change to max_len.
			delta = hidden_state[:, t, :, feat] - hidden_state[:, t-1, :, feat]
			score = np.sum(np.abs(delta))
			if t == length:
				print("|", end="")
			print("%.2f" % score, end=", ")
		print()


	#for i, rhyme in enumerate(selector.common_rhyme):


def visualize_position(seq2seq):
	hidden_state = np.load("hidden.npy")
	#hidden_state = np.zeros([5, 10, 12])
	print("hidden =", hidden_state.shape)  # (8, 20, 1000, 256) -> L = [2, 9]
	#exit()
	
	num_rt = 5
	num_each = 5
	cd = [39, 68, 79, 105, 133, 135, 136, 151, 157, 165, 170, 189, 191, 212, 218, 252]
	#cd = [12, 39, 47, 49, 51, 65, 68, 73, 75, 79, 104, 105, 133, 134, 135, 136, 151, 157, 159, 165, 170, 180, 189, 191, 197, 211, 212, 218, 226, 229, 232, 252]
		
	x = np.empty([num_rt, num_each*hidden_state.shape[2], hidden_state.shape[3]])
	N = hidden_state.shape[2]
	for i in range(0, num_rt):
		for j, jj in enumerate(range(7, 7-num_each, -1)):  # L = [9, 8, 7, 6, 5, 4], class_index = [7, 6, 5, 4, 3, 2], rt starts from 9 
			x[i, j*N:(j+1)*N] = hidden_state[jj, jj-i+2]


	#visualization.scatter_one_plot(x[:, :, cd], reduction_method="PCA", figure_path="", name='Scatter')
	cd = [39, 65, 68, 75, 105, 136, 157, 191, 212, 232, 252]
	for feat in cd[:10]:
		print("feat =", feat, end=": ")
		for i in range(num_rt):
			print("%.2f" % np.mean(x[i, :, feat]), end=", ")
		print()
		plt.plot(list(range(num_rt)), np.mean(x[:num_rt, :, feat], axis=1))
		plt.tight_layout()
	plt.savefig("../../cd.png")
	plt.show()	
	plt.close()


def visualize_value(seq2seq, selector):
	#selector.common_word = [3, 4, 5, 6, 7, 8, 9]
	#conditions = selector.get_word_condition(length="5")
	#sample, sample_index = selector.get_sample(conditions)
	
	weight = seq2seq.decoder_model.get_layer("decoder_gru").get_weights()
	dense_weight = seq2seq.decoder_model.get_layer("output_dense").get_weights()
	
	#pdb.set_trace()
	#hidden = analysis.get_gate_values(seq2seq, sample)
	tmpnp = np.load("hidden_3.npy")
	hidden = {"h": tmpnp[0], "z": tmpnp[1], "r": tmpnp[2], "hh": tmpnp[3]}
		
	import tmp
	import importlib
	pdb.set_trace()
	
	# importlib.reload(tmp)
	tmp.store(hidden, selector, weight, dense_weight)
	for gate in ["h", "hh", "z"]:
		print("gate =", gate)
		for i in selector.fc[:10]:
			print("neuron ", i, end=": ")
			for t in range(10):
				print("%.2f" % np.mean(hidden[gate][4, t, :, i]), end=", ")
			print()
	for gate in ["h", "r"]:
		print("gate =", gate)
		for i in selector.rfe_store[:10]:
			print("neuron ", i, end=": ")
			for t in range(10):
				print("%.2f" % np.mean(hidden[gate][4, t, :, i]), end=", ")
			print()
	print("fc, z")
	for t in range(10):
		print("%.2f" % np.mean(hidden["z"][4, t, :, selector.fc[:16]]), end=", ")
	print()
	print("store, r")
	for t in range(10):
		print("%.2f" % np.mean(hidden["r"][4, t, :, selector.rfe_store[:16]]), end=", ")
	print()
	print("fc, h")
	for t in range(10):
		print("%.2f" % np.mean(np.abs(hidden["h"][4, t, :, selector.fc[:16]])), end=", ")
	print()
	print("store, h")
	for t in range(10):
		print("%.2f" % np.mean(np.abs(hidden["h"][4, t, :, selector.rfe_store[:16]])), end=", ")
	

def visualize_storing(seq2seq):
	
	encoder_hidden_state = np.load("encoder_hidden_word.npy")
	decoder_hidden_state = np.load("hidden_word.npy")
	#other = np.load("hidden_word_token3_other.npy")
	#token3 = np.load("hidden_word_token3.npy")
	#encoder_other = np.load("encoder_hidden_word_token3_other.npy")
	#encoder_token3 = np.load("encoder_hidden_word_token3.npy")
	
	# token 3, 3 to 12, 13 one times, 13 one times -3 dim.
	store_list = [[41, 60, 75, 131, 136, 150, 204, 212, 2, 26, 29, 199, 213, 220, 232, 248, 78, 88, 105, 108, 138, 185, 207, 217, 34, 51, 63, 118, 124, 197, 233, 254, 4, 46, 90, 146, 158, 182, 201, 214, 18, 36, 65, 91, 161, 171, 219, 229, 8, 9, 134, 157, 175, 179, 218, 252, 43, 44, 57, 84, 114, 127, 156, 209, 58, 68, 86, 94, 159, 164, 167, 238, 39, 67, 74, 95, 206, 234, 239, 253, ],
					[63, 8, 60, 254, 233, 118, 217, 238, 105, 207, 41, 18, 199, 146, 171, 138, 220, 78, 75, 70, 185, 136, 209, 51, 57, 134, 88, 159, 248, 197, 108, 91, 58, 68, 213, 214, 150, 180, 7, 5, 29, 89, 9, 235, 4, 139, 26, 204, 212, 167, 73, 252, 37, 49, 124, 253, 115, 15, 2, 65, 34, 13, 230, 232],
					[41, 89, 118, 150, 207, 212, 220, 254, 8, 57, 63, 131, 134, 136, 185, 217, 60, 68, 75, 105, 146, 197, 230, 233, 7, 18, 138, 167, 180, 192, 235, 248, 33, 49, 73, 79, 90, 115, 139, 253, 5, 13, 70, 91, 104, 161, 182, 213, 27, 28, 66, 171, 187, 204, 232, 252, 4, 16, 34, 58, 78, 162, 223, 238],
					[41, 89, 118, 150, 207, 212, 220, 254, 63, 131, 136, 185, 217, 60, 68, 75, 105, 146, 197, 230, 233, 7, 18, 138, 167, 180, 192, 235, 248, 33, 49, 73, 79, 90, 115, 139, 253, 5, 13, 70, 91, 104, 161, 182, 213, 27, 28, 66, 171, 187, 204, 232, 252, 4, 16, 34, 58, 78, 162, 223, 238]
					]
	position = [135, 148, 157, 167, 170, 189, 218, 223, 2, 13, 39, 65, 87, 171, 191, 232, 27, 34, 73, 131, 175, 178, 196, 213, 15, 90, 125, 164, 180, 200, 205, 214, 9, 22, 66, 68, 105, 123, 156, 235, 18, 33, 72, 119, 138, 192, 219, 245, 1, 47, 58, 75, 128, 134, 233, 251, 8, 10, 44, 69, 79, 136, 169, 201, 112, 161, 165, 182, 220, 226, 234, 246, 77, 88, 93, 106, 127, 185, 231, 255, ]
	target_t = 1 + 5   # As length = 5 + 5 encoder time step
	hidden_state = np.concatenate([encoder_hidden_state[:, -5:], decoder_hidden_state[:, 1:]], axis=1)
	
	for index, store in enumerate(store_list[2:3]):
		# Get PCA mapping
		#store = store[:2]
		store = store[:16]
		target = hidden_state[:, target_t]
		target = target[:, :, store]
		target = target.reshape([-1, target.shape[-1]])
		print("target =", target.shape)
		pca = PCA(n_components=2).fit(target)
		words = ["i", "the", "you", "and", "to", "a", "it", "me", "s", "nt"]

		for t in range(13):  # As length = 6
			print("t =", t)
			plt.figure(figsize=(3, 2.25))
			x = hidden_state[:, t, :]
			x = x[:, :, store]
			x_transpose = x.reshape([-1, x.shape[-1]])  # PCA
			embedding = pca.transform(x_transpose)  # PCA
			embedding = embedding.reshape([x.shape[0], x.shape[1], 2])  # Change shape to: (Class, sample, 2).  # PCA
			#embedding = np.copy(x)  # 2-dim
			#plt.axis([-1, 1, -1, 1])  # 2-dim
			for i in range(x.shape[0]):
				plt.scatter(embedding[i, :, 0], embedding[i, :, 1], s=5)
				#plt.scatter([1, 1], [2, 2], s=20, label=words[i])
				#plt.scatter(embedding[i, :, 0], embedding[i, :, 1], s=10, c=color[i], label=label[i])
			plt.tight_layout()
			#plt.legend()
			plt.savefig("../../../figure_1203/store_" + str(index) + "_t=" + str(t-5) + ".png")
			#plt.savefig("../../../figure_1203/store_token.png")
			plt.show()
			plt.close()
			
	
	return
	
	# Plot one class with 10 neurons real values
	# RFE, t = 5, L = 6, class = All, 3, 4, 5, 
	number = 10
	class_ = 0
	#xaxis = ["Token", "EOS", "Word", "Pos.", "SOS", "t=1", "t=2", "t=3", "t=4", "t=5", "t=6"]
	xaxis = ["In-2", "In-1", "EOS", "Word", "Pos.", "t=1", "t=2", "t=3", "t=4", "t=5", "t=6", "t=7"]
	xaxis = ["In", "EOS", "Token", "Pos.", "t=1", "t=2", "t=3", "t=4", '"i"', "t=6"]
	
	hidden_mean = np.mean(hidden_state, axis=2)
	print("x =", hidden_mean.shape)
	plt.figure(figsize=(6, 3))
	for index, store in enumerate(store_list[2:3]):
		
		ax = plt.subplot(111)
		store = store[0:5] + store[6:8] + store[9:10] + store[12:13] + store[14:15]
		#store = store[:10]
		#store = position[:10]
		store.sort()
		x = hidden_mean[class_]
		ax.axvline(x=2, c="black", ls="--")
		ax.axvline(x=8, c="black", ls="--")
		for feat in store:
			x0 = x[:, feat]  # (20)
			#print(x0[-3:T].shape, x0.shape, x.shape, feat, store)
			ax.plot(xaxis, x0[1 : len(xaxis)+1], label="N " + str(feat))
		#ax.tight_layout()
		box = ax.get_position()
		plt.tight_layout()
		ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
		ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # 
		
		
		plt.savefig("../../../figure_1203/store.png")
		plt.show()	
		plt.close()

def verify_position_encoder(seq2seq, selector):
	cd = [12, 39, 47, 49, 51, 65, 68, 73, 75, 79, 104, 105, 133, 134, 135, 136, 151, 157, 159, 165, 170, 180, 189, 191, 197, 211, 212, 218, 226, 229, 232, 252]
	
	
	encoder_hidden_state = np.load("encoder_hidden.npy")
	decoder_hidden_state = np.load("hidden.npy")
	print(encoder_hidden_state.shape, decoder_hidden_state.shape)
	
	
	cd = [39, 68, 79, 105, 133, 135, 136, 151, 157, 165, 170, 189, 191, 212, 218, 252]
	cd = [65, 136, 148, 167, 170, 191, 218, 223, 2, 14, 34, 133, 135, 180, 213, 214, 39, 128, 131, 156, 175, 178, 190, 232, 13, 68, 76, 77, 119, 157, 188, 227, 6, 15, 42, 67, 106, 123, 211, 246, 10, 72, 109, 114, 166, 205, 235, 243, 9, 52, 56, 95, 154, 171, 229, 233, 35, 73, 87, 141, 153, 163, 179, 186, 16, 45, 47, 85, 145, 181, 196, 244, 33, 90, 182, 194, 200, 219, 234, 249, ]
	hidden_state = np.concatenate([encoder_hidden_state[:, -3:], decoder_hidden_state[:, 1:10]], axis=1)  # time
	hidden_state = np.mean(hidden_state, axis=2)  # Mean sample
	#x = 
	hidden_state = hidden_state[:, :, [65, 167, 2, 34, 180, 213, 178, 232, 136, 170, 191, 218, 135, 214, 39, 128, 131, 175]] # cd[:32]
	print(hidden_state.shape)
	#ax.axvline(x=8, c="black", ls="--")
	#print(123)
	#for i in range(hidden_state.shape[2]):
	#	if hidden_state[4, 6, i] - hidden_state[4, 4, i] < 0:
	for i in range(8, hidden_state.shape[2]):
		hidden_state[:, :, i] *= -1
	hidden_state = np.mean(hidden_state, axis=-1)
	print(hidden_state.shape)

	plt.figure(figsize=(6, 3))
	plt.axvline(x=2, c="black", ls="--")
	x = ["EOS", "Word", "Pos.", "t=1", "t=2", "t=3", "t=4", "t=5", "t=6", "t=7", "t=8", "t=9"]
	x = ["EOS", "Token", "Pos.", "t=1", "t=2", "t=3", "t=4", "t=5", "t=6", "t=7", "t=8", "t=9"]
	for i in range(2, hidden_state.shape[0]):
		plt.plot(x, hidden_state[i], label="T=" + str(i+2))
		#plt.scatter(i+3, hidden_state[i, i+3], s=10)
		plt.axvline(x=i+3, C="C" + str(i-2), ls="--")
	#print(y.shape)
	#for i in cd[:10]:
	#	plt.plot(x, y[:, i])
		
	plt.tight_layout()
	plt.legend()
	plt.show()
	


"""