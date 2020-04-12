import numpy as np 

def select_large_weight(weight, target, mode="backward", threshold=0.5, absolute=True):
    """Use percentile to filter larger absolute weight.
    "backward" is to find our which {i-set} can influence neuron j.
    Threshold can be ratio (0 <= threshold <= 1), or the exact count (> 1).
    """
    
    assert mode == "forward" or mode == "backward", "Mode must be forward or backward."
    if len(weight.shape) == 1:
        column = np.copy(weight)
    elif mode == "backward":
        column = weight[:, target]
    else:
        column = weight[target]
        
    if absolute: 
        column = np.abs(column)
    sorted_column = -np.sort(-column)
    argsorted_column = np.argsort(-column)
    if threshold <= 1:
        threshold = int(threshold * column.shape[0])
    return argsorted_column[:threshold]