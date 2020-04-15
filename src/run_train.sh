#!/bin/bash


#python3 ../src/main.py --task=autoenc-last --units=64 --max_epochs=100 --mode=train --data_name=auto-last-toy --batch_size=64 --earlyStop_acc=1.0 

#python3 ../src/main.py --task=autoenc-last --units=64 --max_epochs=25 --mode=train --data_name=auto-last-toy --batch_size=64 --earlyStop_acc=1.0 --enable_earlyStop=false --random_seed=2

#python3 ../src/main.py --task=autoenc-last --units=32 --max_epochs=60 --mode=train --data_name=auto-last-toy --batch_size=64 --earlyStop_acc=1.0 --enable_earlyStop=false --random_seed=2

python3 ../src/main.py --task=token-posi --units=64 --max_epochs=60 --mode=train --data_name=token-posi-toy --batch_size=64 --earlyStop_acc=1.0 --enable_earlyStop=true --random_seed=1