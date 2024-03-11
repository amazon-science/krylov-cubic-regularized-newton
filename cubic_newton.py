import argparse
import sklearn.datasets
import urllib.request
import os.path

import numpy as np
import matplotlib.pyplot as plt

from optimizer.loss import LogisticRegression
from optimizer.cubic import Cubic_LS, Cubic_Krylov_LS, SSCN

import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cubic Regularized Newton Methods')
    parser.add_argument('--dataset', metavar='DATASETS', default='w8a', type=str,
                    help='The dataset')
    parser.add_argument('--plot_time', dest='plot_time', action='store_true',
                    help='Plot with respect to time')
    parser.add_argument('--it_max', default=50000, type=int, metavar='IT',
                    help='max iteration')
    parser.add_argument('--time_max', default=60, type=float, metavar='T',
                    help='max time')
    
    parser.add_argument('--SSCN_dim', nargs='+', default=10, type=int, metavar='D',
                    help='Subspace dimensions of SSCN')
    
    args = parser.parse_args()
    dataset = args.dataset
    plot_time = args.plot_time
    it_max = args.it_max
    time_max = args.time_max

    m_list = args.SSCN_dim
    if isinstance(m_list, int):
        m_list = [m_list]

    # Load LIBSVM datasets
    # dataset = 'gisette_scale'
    # dataset = 'madelon'
    # dataset = 'rcv1_train.binary'
    # dataset = 'news20.binary'
    data_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{}".format(dataset)

    if dataset in {'gisette_scale','duke','rcv1_train.binary','news20.binary'}:
        data_url = '{}.bz2'.format(data_url)
        data_path = './{}.bz2'.format(dataset)
    else:
        data_path = './{}'.format(dataset)
    if not os.path.exists(data_path):
        f = urllib.request.urlretrieve(data_url, data_path)
    A, b = sklearn.datasets.load_svmlight_file(data_path)
    
    # Converting A into the Compressed Sparse Column (CSC) format; this is particularly suitable for SSCN
    A_csc = A.tocsc()

    # Define loss functions
    loss = LogisticRegression(A, b, l1=0, l2=0, store_mat_vec_prod=True)
    loss_csc = LogisticRegression(A_csc, b, l1=0, l2=0, store_mat_vec_prod=True)
    n, dim = A.shape
    x0 = np.ones(dim) * 0.5

    # Define the optimization algs

    # Krylov Cubic Regularized Newton
    memory_size = 10
    cub_krylov = Cubic_Krylov_LS(loss=loss, reg_coef = 1e-3, label='Krylov CRN (m = {})'.format(memory_size),
                            subspace_dim=memory_size, tolerance = 1e-9)
    
    # cub_krylov_bench is used to compute the optimal value of the function
    memory_size_bench = 20
    cub_krylov_bench = Cubic_Krylov_LS(loss=loss, reg_coef= 1e-3, label='Benchmark Krylov CRN (m = {})'.format(memory_size_bench),
                               subspace_dim=memory_size_bench, tolerance = 1e-9)
    
    # Cubic Regularized Newton
    if dim < 500: 
    # When dim is small, directly solve the linear systems of equations in cubic subproblems
        cubic_solver = "full"
    else: 
    # Otherwise, use conjugate gradient method 
        cubic_solver = "CG"
    cub_root = Cubic_LS(loss=loss, reg_coef = 1e-3, label='CRN', cubic_solver=cubic_solver, tolerance = 1e-8)

    # Stochastic Subspace Cubic Newton
    sscn_list = [ ]
    for m in m_list:
        sscn_list.append(SSCN(loss=loss_csc, reg_coef = 1e-3, label='SSCN (m = {})'.format(m),
                            subspace_dim=m, tolerance = 1e-9))

    
    # Running algs
    # Cubic regularized Newton
    print(f'Running optimizer: {cub_root.label}')
    cub_root.run(x0=x0, it_max=it_max, t_max=time_max)
    cub_root.compute_loss_of_iterates()
    time_max = max(cub_root.trace.ts[-1],time_max)

    # SSCN
    for algs in sscn_list:
        print(f'Running optimizer: {algs.label}')
        algs.run(x0=x0, it_max=it_max, t_max=time_max)
        algs.compute_loss_of_iterates()

    # Krylov Cubic Regularized Newton
    print(f'Running optimizer: {cub_krylov.label}')
    cub_krylov.run(x0=x0, it_max=it_max, t_max=time_max)
    cub_krylov.compute_loss_of_iterates()

    print(f'Running optimizer: {cub_krylov_bench.label}')
    cub_krylov_bench.run(x0=x0, it_max=5*it_max, t_max=5*time_max)
    cub_krylov_bench.compute_loss_of_iterates()

    # Plot the loss curve
    # plt.style.use('tableau-colorblind10')
    sns.set_style('ticks') # setting style
    # sns.set_context('paper') # setting context
    sns.set_palette('colorblind') # setting palette

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Avoid Type 3 fonts in the plots
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Ref: https://stackoverflow.com/a/39566040
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



    f_opt = min(loss.f_opt, loss_csc.f_opt)
    cub_root.trace.plot_losses(marker='o', markersize=5, f_opt=f_opt, time=plot_time)
    for algs in sscn_list:
        algs.trace.plot_losses(marker='^', markersize=6, f_opt=f_opt, time=plot_time)
    cub_krylov.trace.plot_losses(marker='v', markersize=6, f_opt=f_opt, time=plot_time, color = color_cycle[7])
    if plot_time:
        plt.xlabel('Time (s)')
    else:
        plt.xlabel('Iteration')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.title('{} ($n={:,}$, $d={:,}$)'.format(dataset,n,dim))

    if not os.path.exists('figs'):
        os.makedirs('figs')

    if plot_time:
        plt.savefig('figs/time_{}.pdf'.format(dataset))
    else:
        plt.savefig('figs/iteration_{}.pdf'.format(dataset))
    # plt.show()