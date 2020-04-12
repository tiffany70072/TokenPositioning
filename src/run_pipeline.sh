#!/bin/bash

#saved_path="../result/neuron_selection_0228/"
saved_path="../result/neuron_selection_0322/"
tokens="3 4 5 6 7 10 25 50 100 250 500 8 9 "
tokens="11 12 13 14 15 16 17 18 19 20 21 22 23 24 26 27 28 29 30"
tokens="3 4 5 6 7 8 9 10 25 50 100"
tokens="11 12 13 14 15 16 17 18 19 20 21 22 23 24"

for token in $tokens; do
    time python3 pipeline_neuron_verification.py $token ${saved_path} > ${saved_path}log_token${token};
    time python3 pipeline_neuron_intersection.py $token ${saved_path};
    
done




