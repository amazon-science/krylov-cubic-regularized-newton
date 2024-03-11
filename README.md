# Krylov Cubic Regularized Newton

**Authors: Ruichen Jiang, Parameswaran Raman, Shoham Sabach, Aryan Mokhtari, Mingyi Hong, Volkan Cevher**

This repository contains the python code for the numerical experiments in our paper [Krylov Cubic Regularized Newton: A Subspace Second-Order Method with Dimension-Free Convergence Rate](https://arxiv.org/abs/2401.03058), which has been accepted by AISTATS 2024.   

## How to Run

To reproduce the plots in Figure 2 of our paper, run the bash script "cubic_newton.sh": 

```
./cubic_newton.sh
```

This calls the python script "cubic_newton.py", which solves a logistic regression problem on LIBSVM datasets (Chang and Lin, 2011) using **Cubic Regularized Newton (CRN)** (Nesterov and Polyak, 2006), **Stochastic Subpsace Cubic Newton (SSCN)** (Hanzely et al., 2020), and our proposed **Krylov Cubic Regularized Newton**. 
The python script accepts the following arguments: 

- `--dataset DATASET`: Specifies the [LIBSVM dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) to test the algorithms. Examples: w8a, rcv1_train.binary, news20.binary
- `--plot_time`: Enables plotting the suboptimality gap against the overall computation time. If not set, plots will show the gap against the number of iterations.
- `--it_max IT`: Sets the maximum number of iterations.
- `--time_max T`: Sets the maximum computation time in seconds.
- `--SSCN_dim D [D ...]`: Specifies a list of subspace dimensions for the SSCN method.

For example, to run an experiment on the "rcv1_train.binary" dataset, compare the performance of algorithms by computation time, with a cap of 50,000 iterations and a 60-second time limit, and test various subspace dimensions for SSCN, use the following command:

```bash
python cubic_newton.py --dataset rcv1_train.binary --plot_time --it_max 50000 --time_max 60 --SSCN_dim 10 50 100 500
```

The generated plots will then be stored in the "./figs" directory.

## Acknowledgment

Our implementation is partially based on [opt_methods](https://github.com/konstmish/opt_methods) by Konstantin Mishchenko.

## Citation

Please consider citing our paper if you use our code:

```text
@article{jiang2024krylov,
  title={Krylov Cubic Regularized Newton: A Subspace Second-Order Method with Dimension-Free Convergence Rate},
  author={Jiang, Ruichen and Raman, Parameswaran and Sabach, Shoham and Mokhtari, Aryan and Hong, Mingyi and Cevher, Volkan},
  journal={arXiv preprint arXiv:2401.03058},
  year={2024}
}
```

## References

C.-C. Chang and C.-J. Lin. LIBSVM: a library for
support vector machines. _ACM Transactions on Intelligent Systems and Technology (TIST)_, 2(3):1–27,
2011.

Y. Nesterov and B. T. Polyak. Cubic regularization of
Newton method and its global performance. _Mathematical Programming_, 108(1):177–205, 2006.

F. Hanzely, N. Doikov, Y. Nesterov, and P. Richtarik.
Stochastic subspace cubic Newton method. In _International Conference on Machine Learning_, pages
4027–4038. PMLR, 2020.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

