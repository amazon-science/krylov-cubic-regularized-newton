#!/bin/bash

python cubic_newton.py --dataset w8a --it_max 100 --time_max 60000
python cubic_newton.py --dataset w8a --plot_time --it_max 50000 --time_max 60
python cubic_newton.py --dataset rcv1_train.binary --it_max 50 --time_max 60000 --SSCN_dim 10 50 100 500
python cubic_newton.py --dataset rcv1_train.binary --plot_time --it_max 50000 --time_max 60 --SSCN_dim 10 50 100 500
python cubic_newton.py --dataset news20.binary --it_max 50 --time_max 60000 --SSCN_dim 10 50 500 1000
python cubic_newton.py --dataset news20.binary --plot_time --it_max 50000 --time_max 60 --SSCN_dim 10 50 500 1000