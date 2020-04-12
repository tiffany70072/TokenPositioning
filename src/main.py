"""Main part of explainable Seq2Seq.
"""


import numpy as np
#import os
import pdb
#import pickle
#from pathlib import Path
from argparse import ArgumentParser
from tensorflow.compat import v1 as tf
from tensorflow.keras import backend as K

#import analysis
#import analyze_weight
#import call_classifier
#import call_integrated_gradient	
#import generate_specific_dataset
#import visualization

from seq2seq import Seq2Seq
from trainer import Trainer 

try:
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.2
	sess = tf.Session(config=config)
	K.set_session(sess)
except AttributeError:
	print("No session. Cannot set session.")
	pass



def set_arguments():
	parser = ArgumentParser()
	
	parser.add_argument("--units", default=256, type=int, help="")
	parser.add_argument("--mode", default="train,evaluate", help="train")
	parser.add_argument("--data_name", default="autoencoder")
	parser.add_argument("--task", default="autoencoder")
	
	# Arguments only for training.
	parser.add_argument("--max_epochs", default=70, type=int, help="")
	parser.add_argument("--batch_size", default=128, type=int)
	parser.add_argument("--earlyStop_acc", default=0.99)
	parser.add_argument("--random_seed", default=1, type=int)
	
	args = parser.parse_args()
	args.model_path = "../saved_model/%s_units=%s_seed=%d" % (args.task, args.units, args.random_seed)
	args.experiment_name = "%s_test" % args.task
	args.log_file = "../result/training_%s_seed=%d.txt" % (args.task, args.random_seed)
	args.mode = args.mode.split(",")
	print("\targs.mode =", args.mode)

	return args


def main():
	args = set_arguments()
	seq2seq = Seq2Seq(args)
	#pdb.set_trace()
	
	if "train" in args.mode:
		trainer = Trainer(args, seq2seq)
		trainer.train()
		#seq2seq.check_accuracy(check_list=["word", "length"])  # Check accuracy after training.
		print("\ttask =", args.task)
		print("\tunits =", args.units)
		print("\t=" * 50)


if __name__ == "__main__":
	main()
	
