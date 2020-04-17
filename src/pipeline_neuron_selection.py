"""
4/15: 
Neuron selection for (1) store (2) counter (3) ig
Neuron verification on these two tasks.
"""


import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import importlib
import os
import pickle
import sys
from sklearn.utils import shuffle
matplotlib.use('GTK')
sys.path.insert(0, '../src/')

import call_classifier
import call_integrated_gradient as call_ig
import evaluator
import neuron_mapping
import sample_getter
import state_getter
import utils 
import verification
from neuron_mapping import evaluate_intersection
from neuron_mapping import get_intersection
from tensorflow.keras.preprocessing.sequence import pad_sequences


def set_arguments():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str,
                        help="{autoencoder, autoenc-last, token-posi}")  # Note: "-", not "_".
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--units", type=int, help="")
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--token", default=3, type=int)
    args = parser.parse_args()
    args.saved_path = os.path.join('../result/', "%s_units=%d_seed=%d" 
                                   % (args.data_name, args.units, args.random_seed))
    args.target_units = units // 2
    return args


def get_store_neuron(token, T_list): 
    print("\tGet Store")
    result = {}
    for T in T_list:
        print("=" * 50 + "\nT = %d" % T)
        si1 = sample_getter.get_sample_by_one_condition(seq2seq.decoder_in_test, 
                                                        token=token, position=T, N=1000)
        si2 = sample_getter.get_sample_by_one_condition(seq2seq.decoder_in_test, 
                                                        token=token, position=T, N=1000, 
                                                        except_this_token=True)    
        sample_index = sample_getter.get_different_amount_sample([si1, si2])
        if sample_index is None or sample_index.shape[1] < 5:
            print("\tToo less samples in this condition.")
            continue  # No any sample for this condition.
        
        result[T] = {}
        state = state_getter.get_hidden_state(seq2seq, sample_index)
        for t in range(T+2):
            print("=" * 50 + "\nt = %d" % t)
            x = state[:, :, t, :]
            y = np.concatenate([np.full([x.shape[1]], 1, dtype=int), np.full([x.shape[1]], 0, dtype=int)])
            x = np.reshape(x, [-1, seq2seq.units])
            x, y = shuffle(x, y, random_state=42)
            features = call_classifier.call_recursive_rfe(x, y, max_count=target_units, one_threshold=0.5)
            result[T][t] = features
            print("features =", features)
    return result


def get_counter_neuron(token, T_list):
    print("\tGet Counter")
    result = {}
    for T in T_list:
        print("=" * 50 + "\nT = %d" % T)
        si = sample_getter.get_sample_by_one_condition(seq2seq.decoder_in_test, 
                                                       token=token, position=T, N=1000)
        sample_index = sample_getter.get_different_amount_sample([si])
        if sample_index is None or sample_index.shape[1] < 5:
            print("\tToo less samples in this condition.")
            return  # No any sample for this condition.
        state = state_getter.get_hidden_state(seq2seq, sample_index)
        state = state[:, :, :T]

        x = state[0].transpose([1, 0, 2])  # [N, t, units] -> [t, N, units]
        y = np.full([x.shape[1]], 0, dtype=int)
        for t in range(1, x.shape[0]):
            y = np.concatenate([y, np.full([x.shape[1]], t, dtype=int)])  
        x = np.reshape(x, [-1, seq2seq.units])
        x, y = shuffle(x, y, random_state=42)
        result[T] = call_classifier.call_recursive_rfe(x, y, max_count=target_units, one_threshold=0.5)
    return result


def get_ig_neuron(token, T_list):
    print("\tGet IG")
    result = {}
    for T in T_list:
        print("=" * 50 + "\nT = %d" % T)
        si1 = sample_getter.get_sample_by_one_condition(seq2seq.decoder_in_test, 
                                                        token=token, position=T, N=1000)
        result[T] = {}
        decoder_states, decoder_inputs = call_ig.get_state_by_sample_index(seq2seq, si1)
        for t in range(T+1):
            print("=" * 50 + "\nt = %d" % t)
            decoder_model = call_ig.get_model_without_argmax(seq2seq, input_t=t, output_t=T)
            score = call_ig.compute_ig_steps(decoder_model, decoder_states[t], decoder_inputs, target_class=token)
            selected = call_ig.get_important_neurons_by_IG(score, k=target_units)
            result[T][t] = selected
            print("\tselected =", selected)
    return result


def verify_original_model(seq2seq, sample, real, token, T):
    print("\tVerify on original model.")
    pred = seq2seq.inference_batch(sample)
    pred = pad_sequences(pred, maxlen=seq2seq.tgt_max_len, padding='post', truncating='post')
    print(pred[:3])
    print("", end="\t")
    evaluator.evaluate_autoencoder_at_time(real, pred, time_step=T, verbose=2)  
    evaluator.evaluate_autoencoder_token(real, pred, token=token)
    
    
def verify_store_one_step(T, t, feature1, feature2, seq2seq, sample, real):
    # Enable and disable store neuron and calculate the accuracy.
    print("T=%d\tt=%d\tN1=%d\tN2=%d" % (T, t, len(feature1), len(feature2)), end="\t")
    evaluate_intersection(feature1, feature2, verbose=2)
    feature = get_intersection(feature1, feature2)
    print("\n\t\t\t", end="")
    pred = verification.verify_decoder(seq2seq, sample, feature, time_step=t, 
                                       mode="disable", replace_by="zero", verbose=2)
    evaluator.evaluate_autoencoder_at_time(real, pred, time_step=T, verbose=2) 
    print("\t\t\t", end="")
    pred = verification.verify_decoder(seq2seq, sample, feature, time_step=t, 
                                       mode="enable", replace_by="zero", verbose=2) 
    evaluator.evaluate_autoencoder_at_time(real, pred, time_step=T, verbose=2) 

    
def verify_store_one_token(token):
    with open(os.path.join(saved_path, 'neuron_token=%d.pickle' % token), 'rb') as handle:
        result = pickle.load(handle)
        store_neuron = result['store']
        ig_neuron = result['ig']
        
    for T in ig_neuron:
        print("-" * 50)
        si1 = sample_getter.get_sample_by_one_condition(seq2seq.decoder_in_test, 
                                                        token=token, position=T, N=100)
        sample = seq2seq.encoder_in_test[si1]
        real = evaluator.get_evaluate_real(seq2seq, si1)
        verify_original_model(seq2seq, sample, real, token, T)
        for t in ig_neuron[T]:
            verify_store_one_step(T, t, store_neuron[T][t], ig_neuron[T][t], seq2seq, sample, real)
    
def verify_counter_one_step(T, t, feature1, feature2, seq2seq, sample, real, token):
    print("T=%d\tt=%d\tN1=%d\tN2=%d" % (T, t, len(feature1), len(feature2)), end="\t")
    neuron_mapping.evaluate_intersection(feature1, feature2, verbose=2)
    feature = neuron_mapping.get_intersection(feature1, feature2)
    print("\n", end="\t")
    pred = verification.verify_decoder(seq2seq, sample, feature, time_step=t, 
                                       mode="disable", replace_by="last_h", verbose=2)
    evaluator.evaluate_autoencoder_at_time(real, pred, time_step=T, verbose=2)  
    evaluator.evaluate_autoencoder_token(real, pred, token=token)
    
    
def verify_counter_one_token(token):
    with open(os.path.join(saved_path, 'neuron_token=%d.pickle' % token), 'rb') as handle:
        result = pickle.load(handle)
        counter_neuron = result['counter']
        ig_neuron = result['ig']
        
    for T in ig_neuron:
        si1 = sample_getter.get_sample_by_one_condition(seq2seq.decoder_in_test, 
                                                        token=token, position=T, N=100)
        sample = seq2seq.encoder_in_test[si1]
        real = evaluator.get_evaluate_real(seq2seq, si1)
        verify_original_model(seq2seq, sample, real, token, T)
        for t in ig_neuron[T]:
            verify_counter_one_step(T, t, counter_neuron[T], ig_neuron[T][t], 
                                    seq2seq, sample, real, token)
        break
        

def enable_important_neuron_one_token(token):
    with open(os.path.join(saved_path, 'neuron_token=%d.pickle' % token), 'rb') as handle:
        result = pickle.load(handle)
        store_neuron = result['store']
        counter_neuron = result['counter']
        ig_neuron = result['ig']
        
    for T in ig_neuron:
        si1 = sample_getter.get_sample_by_one_condition(seq2seq.decoder_in_test, 
                                                        token=token, position=T, N=100)
        sample = seq2seq.encoder_in_test[si1]
        real = evaluator.get_evaluate_real(seq2seq, si1)
        verify_original_model(seq2seq, sample, real, token, T)
        for t in ig_neuron[T]:
            important_store = get_intersection(store_neuron[T][t], ig_neuron[T][t])
            important_counter = get_intersection(counter_neuron[T], ig_neuron[T][t])
            important_neuron = list(set(important_store).union(set(important_counter)))
            print("T=%d, t=%d, (%d, %d, %d)" % (T, t, len(important_neuron), 
                                                len(important_store), len(important_counter)))            
            print("\t", end="")
            pred = verification.verify_decoder(seq2seq, sample, important_neuron, time_step=t, 
                                           mode="enable", replace_by="zero", verbose=2)
            evaluator.evaluate_token(pred, token, T + 3)
            print("\t\t\t", end="")
            evaluator.evaluate_autoencoder_at_time(real, pred, time_step=T, verbose=2)
            #print(pred[:3])
            print("\t", end="")
            pred = verification.verify_decoder(seq2seq, sample, important_neuron, time_step=t, 
                                           mode="disable", replace_by="zero", verbose=2)
            evaluator.evaluate_token(pred, token, T + 3)
            print("\t\t\t", end="")
            evaluator.evaluate_autoencoder_at_time(real, pred, time_step=T, verbose=2)
            #print(pred[:3])




def check_saved_path(saved_path):
    try:
        os.mkdir(saved_path)  # Make directory.
    except FileExistsError:
        pass
    

# Set parameters.
args = set_arguments()
check_saved_path(args.saved_path)
token = args.token
units = args.units
T_list = [5, 7]  # TODO: define by yourself, depends on your data
seq2seq = utils.get_trained_model(args.task, args.data_name, args.units,
                                  random_seed=args.random_seed)
    
# Run neuron selection to get important neurons.
store_neuron = get_store_neuron(token, T_list)
counter_neuron = get_counter_neuron(token, T_list)
ig_neuron = get_ig_neuron(token, T_list)

# Save neuron.
with open(os.path.join(saved_path, 'neuron_token=%d.pickle' % token), 'wb') as handle:
    data = {'store': store_neuron,
           'counter': counter_neuron,
           'ig': ig_neuron}
    pickle.dump(data, handle)
    
# Run neuron verification.
verify_store_one_token(token)
verify_counter_one_token(token)
enable_important_neuron_one_token(token)  # Enable store and counter at the same time.

    
    
 