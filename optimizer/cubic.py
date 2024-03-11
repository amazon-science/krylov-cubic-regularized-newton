# --------------------------------------------------------------------------------
# This file incorporates code from "opt_methods" by Konstantin Mishchenko
# available at https://github.com/konstmish/opt_methods/blob/master/optmethods/second_order/cubic.py
#
# "opt_methods" is licensed under the MIT License. You can find a copy of the license at https://github.com/konstmish/opt_methods/blob/master/LICENSE
# Here is a brief summary of the changes made to the original code:
# - Reimplemented the cubic subproblem solver by solving a 1-d nonlinear equation with Newton's method
# - Added Lanczos method for computing bases of Krylov subspace
# - Reimplemented the cubic regularized Newton method with line search
# - Added Krylov cubic regularized Newton method and stochastic subspace cubic Newton method
#
# --------------------------------------------------------------------------------
import numpy as np
import numpy.linalg as la
import copy

from scipy.optimize import root_scalar
from scipy.linalg import solve 

from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator, spsolve


import random
import os
# import time

from optimizer.optimizer import Optimizer

def set_seed(seed=42):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def cubic_solver_root(g, H, M, it_max=100, epsilon=1e-8, r0 = 0.1):
    """
    Solve min_s <g, s> + 1/2<s, H s> + M/3 ||s||^3
    We follow the implementation in Section 6.1 of
    "Adaptive cubic regularisation methods for unconstrained optimization. Part I: motivation, convergence and numerical results"
    https://link.springer.com/content/pdf/10.1007/s10107-009-0286-5.pdf
    """
    if sparse.issparse(H):
        # when the dimension is small, convert H to a dense matrix
        if len(g) < 500:
            H = H.toarray()
            id_matrix = np.eye(len(g))
            lp_solve = lambda A,b: solve(A,b, assume_a= 'pos')
        else:
            id_matrix = sparse.eye(len(g))
            lp_solve = lambda A,b : spsolve(A,b)
    else:
        id_matrix = np.eye(len(g))
        lp_solve = lambda A,b: solve(A,b, assume_a= 'pos')

    def func(lam):
        s_lam = -lp_solve(H + lam*id_matrix, g)
        return lam**2 - M**2 * np.linalg.norm(s_lam)**2
    
    def grad(lam):
        s_lam = -lp_solve(H + lam*id_matrix, g)
        phi_lam_grad = -2*np.dot(s_lam,lp_solve(H + lam*id_matrix, s_lam))
        return 2*lam - M**2 * phi_lam_grad

    # Solve a 1-d nonlinear equation by Newton's method
    sol = root_scalar(func, fprime=grad, x0 = r0, method='newton', maxiter=it_max, xtol=epsilon)
    r = sol.root
    s = -lp_solve(H + r*id_matrix, g)
    norm_s = la.norm(s)
    model_decrease = r/2*norm_s**2-M/3*norm_s**3 - np.dot(g,s)/2
    return s, sol.iterations, r, model_decrease

def Lanczos(A,v,m=10):
    """
    Lanczos Method. The input A is an operator.
    """
    # initialize beta and v
    beta = 0
    v_pre = np.zeros_like(v)
    # normalize v
    v = v / np.linalg.norm(v)
    # Use V to store the Lanczos vectors
    V = np.zeros((len(v),m))
    V[:,0] = v
    # Use alphas, betas to store the Lanczos parameters
    alphas = np.zeros(m)
    betas = np.zeros(m-1)
    for j in range(m-1):
        w = A(v) - beta * v_pre
        alpha = np.dot(v,w)
        alphas[j] = alpha
        w = w - alpha * v
        beta = np.linalg.norm(w)
        if np.abs(beta) < 1e-6:
            break
        betas[j] = beta
        v_pre = v
        v = w / beta
        V[:,j+1] = v
        
    if m > 1 and j < m-2:
        V = V[:,:j+1]
        alphas = alphas[:j+1]
        betas = betas[:j]
    alphas[-1] = np.dot(v, A(v))
    
    return V, alphas, betas, beta



class Cubic_LS(Optimizer):
    """
    Cubic regularized Newton method with line search.
    The method was studied by Nesterov and Polyak in the following paper:
        "Cubic regularization of Newton method and its global performance"
        https://link.springer.com/article/10.1007/s10107-006-0706-8
    
    Arguments:
        reg_coef (float, optional): an estimate of the Hessian's Lipschitz constant
        cubic_solver: either "CG" or "full". It specifies how to solve the linear system of equations resulting from the cubic subproblem
        solver_it_max: the maximum iterations for solving the cubic subproblem
        solver_eps: the accuracy for solving the cubic subproblem
        beta (float, optional): the backtracking parameter
    """
    def __init__(self, reg_coef=None, cubic_solver="CG", solver_it_max=100, solver_eps=1e-8, beta=0.5, *args, **kwargs):
        super(Cubic_LS, self).__init__(*args, **kwargs)
        self.solver_it = 0
        self.solver_it_max = solver_it_max
        self.solver_eps = solver_eps

        self.beta = beta
        self.r0 = 0.1
        self.residuals = []
        self.value = None

        if reg_coef is None:
            self.reg_coef = self.loss.hessian_lipschitz
        else:
            self.reg_coef = reg_coef
        
        if cubic_solver == "CG":
            self.cubic_solver = self.cubic_solver_root_CG
        elif cubic_solver == "full":
            self.cubic_solver = self.cubic_solver_root_full
        else:
            print("Error: cubic_solver not recognized")
    
    def cubic_solver_root_CG(self, M, it_max=100, epsilon=1e-8, r0 = 0.1):
        """
        Same as cubic_solver_root, but uses conjugate gradient to solve the linear system of equations
        """
        g = self.grad
        def func(lam):
            def mv(v):
                return self.loss.hess_vec_prod(self.x,v) + lam*v
            H_lambda = LinearOperator(shape =(len(g),len(g)), matvec=mv)
            s_lam, exit_code = cg(H_lambda, -g, tol=epsilon)
            return lam**2 - M**2 * np.linalg.norm(s_lam)**2
        
        def grad(lam):
            def mv(v):
                return self.loss.hess_vec_prod(self.x,v) + lam*v
            H_lambda = LinearOperator(shape =(len(g),len(g)), matvec=mv)
            s_lam, exit_code = cg(H_lambda, -g, tol=epsilon)
            Hinv_s_lam, exit_code = cg(H_lambda, s_lam, tol=epsilon)
            phi_lam_grad = -2*np.dot(s_lam, Hinv_s_lam)
            return 2*lam - M**2 * phi_lam_grad

        sol = root_scalar(func, fprime=grad, x0 = r0, method='newton', maxiter=it_max, xtol=epsilon)
        r = sol.root

        def mv(v):
            return self.loss.hess_vec_prod(self.x,v) + r*v
        H_lambda = LinearOperator(shape =(len(g),len(g)), matvec=mv)
        s, exit_code = cg(H_lambda, -g, tol=epsilon)
        norm_s = la.norm(s)
        model_decrease = r/2*norm_s**2-M/3*norm_s**3 - np.dot(g,s)/2
        return s, sol.iterations, r, model_decrease
    
    def cubic_solver_root_full(self, M, it_max=100, epsilon=1e-8, r0 = 0.1):
        g = self.grad
        # id_matrix = np.eye(len(g))
        H = self.hess
        return cubic_solver_root(g, H, M, it_max=it_max, epsilon=epsilon, r0=r0)
    
    def step(self):

        if self.value is None:
            self.value = self.loss.value(self.x)
        
        self.grad = self.loss.gradient(self.x)

        if self.cubic_solver == self.cubic_solver_root_full:
            self.hess = self.loss.hessian(self.x)

        # Terminate if the gradient norm is small
        if np.linalg.norm(self.grad) < self.tolerance:
            return
        # Set the initial value of the regularization coefficient
        reg_coef = self.reg_coef*self.beta

        # Solve the cubic subproblem
        s_new, solver_it, r0_new, model_decrease = self.cubic_solver( 
        reg_coef, self.solver_it_max, self.solver_eps, r0 = self.r0)

        x_new = self.x + s_new
        value_new = self.loss.value(x_new)

        # Backtracking line search
        while value_new > self.value - model_decrease:
            # If the sufficient decrease condition is not satisfied, then we increase the regularization parameter
            reg_coef = reg_coef/self.beta
            s_new, solver_it, r0_new, model_decrease = self.cubic_solver( 
            reg_coef, self.solver_it_max, self.solver_eps, r0 = self.r0)
            x_new = self.x + s_new
            value_new = self.loss.value(x_new)
        self.x = x_new
        self.reg_coef = reg_coef
        self.value = value_new
        self.r0 = r0_new
        
        self.solver_it += solver_it
        
    def init_run(self, *args, **kwargs):
        super(Cubic_LS, self).init_run(*args, **kwargs)
        self.trace.solver_its = [0]
        self.loss.reset()
        
    def update_trace(self):
        super(Cubic_LS, self).update_trace()
        self.trace.solver_its.append(self.solver_it)


class Cubic_Krylov_LS(Optimizer):
    """
    Krylov cubic regularized Newton method with line search
    
    Arguments:
        reg_coef (float, optional): an estimate of the Hessian's Lipschitz constant
        subspace_dim (int, optional): The dimension of the Krylov subspace
        solver_eps: the accuracy for solving the cubic subproblem
        beta (float, optional): the backtracking parameter
    """
    def __init__(self, reg_coef=None, subspace_dim=100, solver_eps=1e-8, beta=0.5, *args, **kwargs):
        super(Cubic_Krylov_LS, self).__init__(*args, **kwargs)    
        self.solver_it = 0
        self.subspace_dim = subspace_dim
        self.solver_eps = solver_eps

        self.beta = beta
        self.r0 = 0.1
        # self.residuals = []
        self.value = None

        if reg_coef is None:
            self.reg_coef = self.loss.hessian_lipschitz
        else:
            self.reg_coef = reg_coef
    
    
    def step(self):

        if self.value is None:
            self.value = self.loss.value(self.x)
        
        self.grad = self.loss.gradient(self.x)
   
        # We access Hessian via the Hessian vector product
        self.hess = lambda v: self.loss.hess_vec_prod(self.x,v)
        # Use Lanczos method to compute an orthogonal basis for the Krylov subspace
        V, alphas, betas, beta = Lanczos(self.hess, self.grad, m=self.subspace_dim)

        # The subspace Hessian
        self.hess = np.diag(alphas) + np.diag(betas, -1) + np.diag(betas, 1)

        # The subspace gradient
        e1 = np.zeros(len(alphas))
        e1[0] = 1
        self.grad = np.linalg.norm(self.grad)*e1

        # set the initial value of the regularization coefficient
        reg_coef = self.reg_coef*self.beta

        # Solve the cubic subproblem over the subspace
        s_new, solver_it, r0_new, model_decrease = cubic_solver_root(self.grad, self.hess, 
        reg_coef, epsilon = self.solver_eps, r0 = self.r0)
        x_new = self.x + V @ s_new
        value_new = self.loss.value(x_new)

        iter_count = 0
        max_iter = 20
        # Backtracking line search
        while value_new > self.value - model_decrease and iter_count < max_iter:
            reg_coef = reg_coef/self.beta
            s_new, solver_it, r0_new, model_decrease = cubic_solver_root(self.grad, self.hess, 
            reg_coef, epsilon = self.solver_eps, r0 = self.r0)
            x_new = self.x + V @ s_new
            value_new = self.loss.value(x_new)
            iter_count += 1
        self.x = x_new
        self.reg_coef = reg_coef
        self.value = value_new
        self.r0 = r0_new
        
        self.solver_it += solver_it

        
    def init_run(self, *args, **kwargs):
        super(Cubic_Krylov_LS, self).init_run(*args, **kwargs)
        self.trace.solver_its = [0]
        self.loss.reset()
        
    def update_trace(self):
        super(Cubic_Krylov_LS, self).update_trace()
        self.trace.solver_its.append(self.solver_it)

class SSCN(Optimizer):
    """
    Stochastic Subspace Cubic Newton. This is proposed in the following paper
        "Stochastic subspace cubic Newton method"
        https://proceedings.mlr.press/v119/hanzely20a/hanzely20a.pdf
    In particular, we implement the coordinate version as discussed in Section 7.1
    
    Arguments:
        reg_coef (float, optional): an estimate of the Hessian's Lipschitz constant
        subspace_dim (int, optional): the dimension of the random subspace
        solver_eps: the accuracy for solving the cubic subproblem
        beta (float, optional): the backtracking parameter
    """
    def __init__(self, reg_coef=None, subspace_dim=100, solver_eps=1e-8, beta=0.5, *args, **kwargs):
        super(SSCN, self).__init__(*args, **kwargs)
        self.reg_coef = reg_coef
        self.solver_it = 0
        self.subspace_dim = subspace_dim
        self.solver_eps = solver_eps

        self.beta = beta
        self.r0 = 0.1
        self.residuals = []
        self.value = None
        self.tolerance = 0

        if reg_coef is None:
            self.reg_coef = self.loss.hessian_lipschitz

        self.reuse = False
    
    def step(self):
        if self.value is None:
            self.value = self.loss.value(self.x)
        
        # sample random coordinates
        I = self.rng.choice(self.dim, size=self.subspace_dim, replace=False)
        
        # compute coordinate gradient
        self.grad = self.loss.partial_gradient(self.x, I)

        # compute coordinate Hessian
        self.hess = self.loss.partial_hessian(self.x, I)

        # set the initial value of the regularization coefficient
        reg_coef = max(self.reg_coef*self.beta, np.finfo(float).eps)


        # set x_new to be self.x
        x_new = copy.deepcopy(self.x)
        Ax = copy.deepcopy(self.loss._mat_vec_prod)

        s_new_sub, solver_it, r0_new, model_decrease = cubic_solver_root(self.grad, self.hess, 
        reg_coef, r0 = self.r0, epsilon=np.finfo(float).eps)
        # update the chosen coordinates in x_new
        x_new[I] = self.x[I] + s_new_sub

        # update Ax in the memory
        self.loss.update_mat_vec_product(Ax, s_new_sub, I)
        
        value_new = self.loss.value(x_new)

        # Backtracking line search
        while value_new > self.value - model_decrease:
            reg_coef = reg_coef/self.beta
            s_new_sub, solver_it, r0_new, model_decrease = cubic_solver_root(self.grad, self.hess, 
            reg_coef, r0 = self.r0, epsilon=np.finfo(float).eps)
            x_new[I] = self.x[I] + s_new_sub

            # update Ax in the memory
            self.loss.update_mat_vec_product(Ax, s_new_sub, I)
            value_new = self.loss.value(x_new)
        self.x = x_new
        self.reg_coef = reg_coef
        self.value = value_new
        self.r0 = r0_new

        self.solver_it += solver_it
        # self.residuals.append(residual)
        
    def init_run(self, *args, **kwargs):
        super(SSCN, self).init_run(*args, **kwargs)
        self.trace.solver_its = [0]
        self.loss.reset()
        
    def update_trace(self):
        super(SSCN, self).update_trace()
        self.trace.solver_its.append(self.solver_it)