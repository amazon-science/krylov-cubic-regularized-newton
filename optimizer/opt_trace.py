# --------------------------------------------------------------------------------
# This file incorporates code from "opt_methods" by Konstantin Mishchenko
# available at https://github.com/konstmish/opt_methods/blob/master/optmethods/opt_trace.py
#
# "opt_methods" is licensed under the MIT License. You can find a copy of the license at https://github.com/konstmish/opt_methods/blob/master/LICENSE
# Here is a brief summary of the changes made to the original code:
# - Removed the class "StochasticTrace"
# 
# --------------------------------------------------------------------------------
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle
import warnings


class Trace:
    """
    Stores the logs of running an optimization method
    and plots the trajectory.
    
    Arguments:
        loss (Oracle): the optimized loss class
        label (string, optional): label for convergence plots (default: None)
    """
    def __init__(self, loss, label=None):
        self.loss = loss
        self.label = label
        
        self.xs = []
        self.ts = []
        self.its = []
        self.loss_vals = []
        self.its_converted_to_epochs = False
        self.ls_its = None
    
    def compute_loss_of_iterates(self):
        if len(self.loss_vals) == 0:
            self.loss_vals = np.asarray([self.loss.value(x) for x in self.xs])
        else:
            warnings.warn('Loss values have already been computed. Set .loss_vals = [] to recompute.')
    
    def convert_its_to_epochs(self, batch_size=1):
        if self.its_converted_to_epochs:
            warnings.warn('The iteration count has already been converted to epochs.')
            return
        its_per_epoch = self.loss.n / batch_size
        self.its = np.asarray(self.its) / its_per_epoch
        self.its_converted_to_epochs = True
          
    def plot_losses(self, its=None, f_opt=None, label=None, markevery=None, use_ls_its=True, time=False, *args, **kwargs):
        if label is None:
            label = self.label
        if its is None:
            if use_ls_its and self.ls_its is not None:
                print(f'Line search iteration counter is used for plotting {label}')
                its = self.ls_its
            elif time:
                its = self.ts
            else:
                its = self.its
        if len(self.loss_vals) == 0:
            self.compute_loss_of_iterates()
        if f_opt is None:
            f_opt = self.loss.f_opt
        if markevery is None:
            markevery = max(1, len(self.loss_vals)//20)
        
        plt.plot(its, self.loss_vals - f_opt, label=label, markevery=markevery, *args, **kwargs)
        plt.ylabel(r'$f(x)-f^*$')
        
    def plot_distances(self, its=None, x_opt=None, label=None, markevery=None, use_ls_its=True, time=False, *args, **kwargs):
        if its is None:
            if use_ls_its and self.ls_its is not None:
                its = self.ls_its
            elif time:
                its = self.ts
            else:
                its = self.its
        if x_opt is None:
            if self.loss.x_opt is None:
                x_opt = self.xs[-1]
            else:
                x_opt = self.loss.x_opt
        if label is None:
            label = self.label
        if markevery is None:
            markevery = max(1, len(self.xs)//20)
            
        dists = [self.loss.norm(x-x_opt)**2 for x in self.xs]
        plt.plot(its, dists, label=label, markevery=markevery, *args, **kwargs)
        plt.ylabel(r'$\Vert x-x^*\Vert^2$')
        
    @property
    def best_loss_value(self):
        if len(self.loss_vals) == 0:
            self.compute_loss_of_iterates()
        return np.min(self.loss_vals)
        
    def save(self, file_name, path='./results/'):
        # To make the dumped file smaller, remove the loss
        loss_ref_copy = self.loss
        self.loss = None
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(path + file_name, 'wb') as f:
            pickle.dump(self, f)
        self.loss = loss_ref_copy
        
    @classmethod
    def from_pickle(self, path, loss=None):
        if not os.path.isfile(path):
            return None
        with open(path, 'rb') as f:
            trace = pickle.load(f)
            trace.loss = loss
        if loss is not None:
            loss.f_opt = min(self.best_loss_value, loss.f_opt)
        return trace