# Test neurons selection pipeline on all T, t as token = 3.
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
import sample_getter
import state_getter
import utils 
import verification
from neuron_mapping import evaluate_intersection
from neuron_mapping import get_intersection


saved_path = sys.argv[2]  # '../result/neuron_selection_0308'
token = int(sys.argv[1])


def save_neuron_for_one_token(token, T_list=[1, 2, 3, 4, 5, 6, 7]):
    print("Token %d (%s)" % (token, seq2seq.tgt_itoc[token]))
    result1 = {}
    result2 = {}
    for T in T_list:
        print("=" * 50 + "\nT = %d" % T)
        si1 = sample_getter.get_sample_by_one_condition(seq2seq.decoder_in_test, 
                                                        token=token, position=T, N=1000)
        si2 = sample_getter.get_sample_by_one_condition(seq2seq.decoder_in_test, 
                                                        token=token, position=T, N=1000, 
                                                        except_this_token=True)    
        #sample_index = sample_getter.get_same_amount_sample([si1, si2])
        sample_index = sample_getter.get_different_amount_sample([si1, si2])
        if sample_index is None or sample_index.shape[1] < 5:
            print("\tToo less samples in this condition.")
            continue  # No any sample for this condition.
        result1[T] = {}
        result2[T] = {}
        state = state_getter.get_hidden_state(seq2seq, sample_index)
        
        # Call classifier.
        for t in range(T+2):
            print("=" * 50 + "\nt = %d" % t)
            x = state[:, :, t, :]
            y = np.concatenate([np.full([x.shape[1]], 1, dtype=int), np.full([x.shape[1]], 0, dtype=int)])
            x = np.reshape(x, [-1, seq2seq.units])
            x, y = shuffle(x, y, random_state=42)
            features = call_classifier.call_recursive_rfe(x, y, max_count=128, one_threshold=0.5)
            result1[T][t] = features
            print("features =", features)

        # Call IG.
        decoder_states, decoder_inputs = call_ig.get_state_by_sample_index(seq2seq, si1)
        for t in range(T+1):
            print("=" * 50 + "\nt = %d" % t)
            input_t = t
            decoder_model = call_ig.get_model_without_argmax(seq2seq, input_t=input_t, output_t=T)
            score = call_ig.compute_ig_steps(decoder_model, decoder_states[input_t], decoder_inputs, target_class=token)
            selected = call_ig.get_important_neurons_by_IG(score, k=128)
            result2[T][t] = selected
            print("\tselected =", selected)
    print("Result1 =", result1)
    print("Result2 =", result2)

    with open(os.path.join(saved_path, 'neuron_token%s.pickle' % token), 'wb') as handle:
        pickle.dump([result1, result2] , handle)
        
    
def verify_one_time_step(T, t, feature1, feature2, seq2seq, sample, real):
    print("T=%d\tt=%d\tN1=%d\tN2=%d" % (T, t, len(feature1), len(feature2)), end="\t")
    evaluate_intersection(feature1, feature2, verbose=2)
    feature = get_intersection(feature1, feature2)

    pred = verification.verify_decoder(seq2seq, sample, feature, time_step=t, 
                                       mode="disable", replace_by="zero", verbose=2)
    evaluator.evaluate_autoencoder_at_time(real, pred, time_step=T, verbose=2)  # T
    print("\t" * 10, end="\t")
    pred = verification.verify_decoder(seq2seq, sample, feature, time_step=t, 
                                       mode="enable", replace_by="zero", verbose=2) 
    evaluator.evaluate_autoencoder_at_time(real, pred, time_step=T, verbose=2) 


def verify_one_token(token):
    with open(os.path.join(saved_path, 'neuron_token%s.pickle' % token), 'rb') as handle:
        result = pickle.load(handle)
        result1 = result[0]
        result2 = result[1]
        
    for T in result2:
        print("-" * 50)
        si1 = sample_getter.get_sample_by_one_condition(seq2seq.decoder_in_test, 
                                                        token=token, position=T, N=1000)
        sample = seq2seq.encoder_in_test[si1]
        real = evaluator.get_evaluate_real(seq2seq, si1)
        for t in result2[T]:
            verify_one_time_step(T, t, result1[T][t], result2[T][t], seq2seq, sample, real)

    print("\n" * 3)
    for T in result2:
        print("-" * 50)
        si1 = sample_getter.get_sample_by_one_condition(seq2seq.decoder_in_test, token=token, 
                                                        position=T, N=1000)
        sample = seq2seq.encoder_in_test[si1]
        real = evaluator.get_evaluate_real(seq2seq, si1)        
        for t in result2[T]:
            verify_one_time_step(T, t, result1[T][t][:96], result2[T][t][:96], seq2seq, sample, real)

        

seq2seq = utils.get_trained_model(random_seed=3)
save_neuron_for_one_token(token=token, T_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
verify_one_token(token=token)

