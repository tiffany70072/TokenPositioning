import numpy as np 

def set_decoder_output_data(decoder_input):
	decoder_output = decoder_input.copy()
	for i in range(len(decoder_output)): 
		decoder_output[i, :-1] = decoder_input[i, 1:]  # Remove the first token in decoder output.
		decoder_output[i, -1] *= 0
	decoder_output = np.reshape(decoder_output, [decoder_output.shape[0], decoder_output.shape[1], 1])
	return decoder_output

"""
def cut_validation(self):
		# TODO: cut training, validation and testing
		split_result = data_reader.data_split(self.encoder_in, self.decoder_in, self.decoder_out)
		self.encoder_in = split_result[0]
		self.decoder_in = split_result[1]
		self.decoder_out = split_result[2]
		self.encoder_in_valid = split_result[3][:50000]  # TODO: Deal with too many data.
		self.decoder_in_valid = split_result[4][:50000]
		self.decoder_out_valid = split_result[5][:50000]
		self.encoder_in_test = split_result[6]
		self.decoder_in_test = split_result[7]
		self.decoder_out_test = split_result[8]

		self.encoder_in = split_result[0]#[:3000]
		self.decoder_in = split_result[1]#[:3000]
		self.decoder_out = split_result[2]#[:3000]

		print("(Cut validation) training size:", self.encoder_in.shape)
		print("(Cut validation) validation size:", self.encoder_in_valid.shape)
		print("(Cut validation) testing size:", self.encoder_in_test.shape)
"""