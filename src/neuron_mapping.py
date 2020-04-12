import numpy as np 
import os
import pickle


def evaluate_intersection(list1, list2, verbose=1, mode="jaccard", union_len=256, with_count=False):
    list1 = set(list1)
    list2 = set(list2)
    count = len(list1.intersection(list2))
    if mode == "jaccard":
        score = count / float(len(list1.union(list2)))
    elif mode == "smc":  # https://zh.wikipedia.org/wiki/%E7%AE%80%E5%8D%95%E5%8C%B9%E9%85%8D%E7%B3%BB%E6%95%B0
        diff1 = len(list1) - count
        diff2 = len(list2) - count
        score = (union_len - diff1 - diff2) / float(union_len)
    elif mode == "dice":  # https://zh.wikipedia.org/wiki/Dice%E7%B3%BB%E6%95%B0
        score = 2 * count / float(len(list1) + len(list2))
    else:
        print("No this mode.")
        return
    
    
    if verbose == 1:  # For usual output on the terminal.
        print("count=%d, sim=%.2f" % (count, score))
    elif verbose == 2:  # Output in log.
        print("N=%d\tjac=%.2f" % (count, score), end="\t")
    if with_count:
        return score, count
    else:
        return score


def get_intersection(list1, list2):
    list1 = set(list1)
    list2 = set(list2)
    return list(list1.intersection(list2))


# Some utils for experiment 4.
def filter_intersection(list1, list2):
    list1 = set(list1)
    list2 = set(list2)
    intersection = list1.intersection(list2)
    diff1 = list1 - intersection
    diff2 = list2 - intersection
    print("Filter: intersection=%d, diff1=%d, diff2=%d" % (len(intersection), len(diff1), len(diff2)))
    return list(intersection), list(diff1), list(diff2)
    

def get_all_abs_weight(U):
    weights = np.abs(np.ndarray.flatten(U))
    return weights


def get_subset_abs_weight(U, i_set, j_set):
    tmp = np.take(U, i_set, axis=0)
    tmp = np.take(tmp, j_set, axis=1)
    tmp = np.abs(np.ndarray.flatten(tmp))
    return tmp


def get_stable_neuron(state, neuron, t1, t2, threshold=0.05):
    # layer2 = layer1 + 1, list1(2) are the important neurons in layer1(2).
    assert t2 == t1 + 1, "Only support adjacent layers."
    intersection = get_intersection(neuron[t1], neuron[t2])
    h1 = np.mean(state[0, :, t1], axis=0)
    h2 = np.mean(state[0, :, t2], axis=0)
    container = []
    for index in intersection:
        if np.abs(h2[index] - h1[index]) <= threshold:
            container.append(index)
    print("\t#(stable) = %d / #(intersec) = %d" % (len(container), len(intersection)), end="\t")
    return container


def test_get_stable_neuron(token=3, T=3):
    # For token = 3, T = 3
    import sample_getter
    import state_getter
    

    si1 = sample_getter.get_sample_by_one_condition(seq2seq.decoder_in_test, 
                                                    token=token, position=T, N=10)
    sample_index = sample_getter.get_same_amount_sample([si1])
    state = state_getter.get_hidden_state(seq2seq, sample_index)
    for t1 in result3[T]:  
        t2 = t1 + 1
        if t2 not in result3[T]: break  
        for threshold in [0.0, 0.01, 0.05, 0.1, 0.5]:
            get_stable_neuron(state, result3[T], t1, t2, threshold=threshold)
        print("-" * 10)
        
        
        
def get_intersection_by_order(result1, result2):
    result = []
    for i in result2:
        if i in result1:
            result.append(i)
    return result


def get_neuron_from_pickle(saved_path, token):
    pickle_file = os.path.join(saved_path, "neuron_token%s.pickle" % token)
    with open(pickle_file, 'rb') as handle:
        result = pickle.load(handle)
        result1 = result[0]
        result2 = result[1]

    result3 = {}
    for T in result2:
        result3[T] = {}
        for t in result2[T]:
            #feature = get_intersection(result1[T][t], result2[T][t])
            feature = get_intersection_by_order(result1[T][t], result2[T][t])
            result3[T][t] = feature
    return result3


def get_intersection_by_multiple_list(list_list, verbose=1):
    result = set(list_list[0])
    for list_ in list_list[1:]:
        if verbose: print(len(result), "->", end=" ")
        result = result.intersection(set(list_))
    if verbose: print(len(result))
    return list(result)


def get_common_by_multiple_list(list_list, ratio=0.5, unit=256):
    counter = np.zeros([unit])
    for list_ in list_list:
        for x in list_:
            counter[x] += 1
    result = []
    if ratio < 1:
        ratio = ratio * len(list_list)
    for i in range(unit):
        if counter[i] >= ratio:
            result.append(i)
    print("get_common_by_multiple_list", len(result))
    return result
        

