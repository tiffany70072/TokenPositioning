import os
import numpy as np 


def trace_weight(model, layer_list=[]):
	# layer_list can contain "decoder_gru" or "output_dense"
	result = []
	for k, layer_name in enumerate(layer_list):	
		weight = model.get_layer(layer_name).get_weights()
		result.append(weight)
	return result


def output_array(arr, name=None):
	if name is not None:
		print(name)

	if len(arr.shape) == 2:
		for i in range(arr.shape[0]):
			for j in range(arr.shape[1]):
				print("%.2f" % arr[i][j], end=",")
			print()
	elif len(arr.shape) == 1:
		for i in range(arr.shape[0]):
			print("%.2f" % arr[i], end=",")
			if i % 10 == 9:
				print()
		print()


def main():
	trace_weight(seq2seq, args.model_path, layer_list=["decoder_gru", "output_dense"])