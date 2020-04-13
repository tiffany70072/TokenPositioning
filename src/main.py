"""Main part of explainable Seq2Seq.
"""


import numpy as np
#import os
import pdb
#from pathlib import Path
from argparse import ArgumentParser
from tensorflow.compat import v1 as tf
from tensorflow.keras import backend as K

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
	parser.add_argument("--task", default="autoencoder", help="{autoencoder, autoenc-last}")  # Note: "-", not "_".
	
	# Arguments only for training.
	parser.add_argument("--max_epochs", default=70, type=int, help="")
	parser.add_argument("--batch_size", default=128, type=int)
	parser.add_argument("--earlyStop_acc", default=0.99)
	parser.add_argument("--random_seed", default=1, type=int)
	
	args = parser.parse_args()
	args.setting_name = "%s_units=%s_seed=%d" % (args.data_name, args.units, args.random_seed)
	args.model_path = "../saved_model/%s" % (args.setting_name)
	args.log_file = "../result/training_%s.txt" % (args.setting_name)
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
		print("\ttask =", args.task)
		print("\tunits =", args.units)
		print("\t=" * 50)
	pdb.set_trace()

    
if __name__ == "__main__":
	main()
	

	
