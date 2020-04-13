import numpy as np 
import pdb
import random
from math import ceil
from numpy.random import seed
from tensorflow.compat.v1 import set_random_seed

from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping


import utils
import evaluator


class Trainer(object):
	
	def __init__(self, args, seq2seq):
		self.batch_size = args.batch_size
		self.max_epochs = args.max_epochs
		self.log_file = args.log_file
		self.earlyStop_acc = args.earlyStop_acc
		self.seq2seq = seq2seq

		seed(args.random_seed)
		random.seed(args.random_seed)
		set_random_seed(args.random_seed)

		with open(self.log_file, "a") as f:
			f.write("=" * 50 + "\n")
			f.write("task = %s\n" % args.task)
			f.write("data = %s\n" % args.data_name)
			f.write("units = %.d\n" % args.units)


	def train(self): 
		"""Train Seq2Seq model."""
		print("\n\nStart training")
		print("\t(Training) train data =", self.seq2seq.encoder_in.shape, self.seq2seq.decoder_in.shape, self.seq2seq.decoder_out.shape)
		print("\t(Training) valid data =", self.seq2seq.encoder_in_valid.shape, self.seq2seq.decoder_in_valid.shape, self.seq2seq.decoder_out_valid.shape)

		# adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, clipvalue=5.0) # amsgrad=False, 
		self.seq2seq.seq2seq_model.compile(optimizer="adam",
								loss=utils.masked_perplexity_loss, 
								metrics=[utils.whole_accuracy, utils.each_accuracy])
		steps_per_epoch = ceil(self.seq2seq.decoder_out.shape[0] / self.batch_size)
		validation_steps = ceil(self.seq2seq.decoder_out_valid.shape[0] / self.batch_size)
        
		earlyStop = [EarlyStopping(monitor="val_each_accuracy", patience=1, verbose=2),]
		if self.seq2seq.decoder_out.shape[0] > 300000:
			print("Too many training data. Limit iteration numbers in one epoch.")
			steps_per_epoch = ceil(300000 / self.batch_size)
		print("Will save at ", self.seq2seq.model_path)
		
		for epoch in range(self.max_epochs):
			history = self.seq2seq.seq2seq_model.fit_generator(
				self.generate_batch_data(self.seq2seq.encoder_in, self.seq2seq.decoder_in, self.seq2seq.decoder_out),
				steps_per_epoch=steps_per_epoch,
				validation_data=self.generate_batch_data(self.seq2seq.encoder_in_valid, self.seq2seq.decoder_in_valid, self.seq2seq.decoder_out_valid),
				validation_steps=validation_steps, 
				callbacks=earlyStop)  # Default epochs = 1 in keras.
			
			self.seq2seq.save_seq2seq()
			self.trace_history(epoch, history)
			if (history.history["val_whole_accuracy"][0] == 1.0 and history.history["val_each_accuracy"][0] == 1.0) or history.history["val_whole_accuracy"][0] >= self.earlyStop_acc: 
				break


	def generate_batch_data(self, encoder_in, decoder_in, decoder_out):
		bs = self.batch_size
		data_num = decoder_out.shape[0]
		loopcount = data_num // bs
		sampling_container = np.ones((bs, 1))
		while True:
			i = random.randint(0, loopcount-1)
			batch_index = [j for j in range(i*bs, (i+1)*bs)]
			
			if random.uniform(0.0, 1.0) > 0.5:
				sampling_ratio = sampling_container
			else:
				sampling_ratio = sampling_container * 0
			yield [encoder_in[batch_index], decoder_in[batch_index], sampling_ratio], decoder_out[batch_index]
			

	def trace_history(self, epoch, history):
		if "autoencoder" in self.seq2seq.task:
			#import test_autoencoder
			whole_accuracy, each_accuracy = evaluator.evaluate_autoencoder(seq2seq=self.seq2seq)
			with open(self.log_file, "a") as fout:
				fout.write("epochs = %d,\t" % epoch)
				fout.write("loss = %.2f,\t" % history.history["loss"][0])
				fout.write("val_loss = %.2f,\t" % history.history["val_loss"][0])
				fout.write("val_each_acc = %.4f,\t" % history.history["val_each_accuracy"][0])
				fout.write("all = %.4f,\n" % history.history["val_whole_accuracy"][0])
				#fout.write("acc = %.2f,\t" % accuracy)
				#fout.write("xh = %.2f, %.2f, %.2f\n" % (xh_acc[0], xh_acc[1], xh_acc[2]))
		
		if self.seq2seq.task == "autoenc-last":
			whole_accuracy, each_accuracy = evaluator.evaluate_autoencoder(seq2seq=self.seq2seq)
			real, pred = evaluator.get_default_sample_from_seq2seq(self.seq2seq)
			last_word_accuracy = evaluator.evaluate_autoencoder_last_step(real, pred)
			
			with open(self.log_file, "a") as fout:
				fout.write("epochs = %d,\t" % epoch)
				fout.write("loss = %.2f,\t" % history.history["loss"][0])
				fout.write("val_loss = %.2f,\t" % history.history["val_loss"][0])
				fout.write("val_each_acc = %.4f,\t" % history.history["val_each_accuracy"][0])
				fout.write("all = %.4f,\t" % history.history["val_whole_accuracy"][0])
				fout.write("last = %.4f\n" % last_word_accuracy)
				