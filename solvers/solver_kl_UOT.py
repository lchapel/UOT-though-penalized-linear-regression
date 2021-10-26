import numpy as np
import scipy as sp


def get_X_lasso(n,m):
    """ Construction of design matrix H

    :param n: int
        Dimension of source histogram
    :param m: int
        Dimension of target histogram
    :return:
        Design (or dictionary) matrix H
    """
    jHa = np.arange(m*n)
    iHa = np.repeat(np.arange(n), m)
    jHb = np.arange(m*n)
    iHb = np.tile(np.arange(m), n) + n
    j = np.concatenate((jHa, jHb))
    i = np.concatenate((iHa, iHb))
    H = sp.sparse.csc_matrix((np.ones(n*m*2), (i, j)), shape=(n+m, n*m))
    return H


def ot_ukl_solve_mu(C, a, b, reg, nitermax=100000, tol=1e-15, P0=None, verbose=False):
    """ Majorization-Minimization algorithm for KL-penalized UOT

    :param C: array, shape(n,m)
        Transport cost
    :param a: array, shape(n,)
        Histogram of source distribution
    :param b: array, shape(m,)
        Histogram of target distribution
    :param reg: float
        Penalization coefficient in UOT formulation
    :param nitermax: int, optional
        Maximum number of iterations
    :param tol: float, optional
        The iteration stops when |P - P_old| < tol
    :param P0: array, shape(n,m), optional
        Initialization of transport plan P
    :param verbose: bool, optional
        If verbose is true, print the error at each iteration
    :return:
        The transport plan w.r.t. the value of reg
    """

    K = np.exp(-C/reg/2)
    if P0 is None:
        P = a[:, None]*b[None, :]
    else:
        P = P0
    
    for i in range(nitermax):
        Pold = P.copy()
        u = np.sqrt(a/(P.sum(1)+1e-16))
        v = np.sqrt(b/(P.sum(0)+1e-16))
        P = np.einsum('ij,ij,i,j->ij', P, K, u, v)

        if verbose:
            print(np.linalg.norm(P-Pold))
        if np.linalg.norm(P-Pold) < tol:
            break
    return P


def KL_divergence(x, y):
    # Compute the Kullback-Leibler divergence between x and y
    return np.sum(x * np.log(x/y+1e-16) - x + y)


def ot_ukl_solve_BFGS(C, a, b, reg, nitermax=100000, tol=1e-14):
    """ BFGS algorithm for KL-penalized UOT

    :param C: array, shape(n,m)
        Cost matrix
    :param a: array, shape(n,)
        Histogram of source distribution
    :param b: array, shape(m,)
        Histogram of target distribution
    :param reg: float
        Penalization coefficient in UOT formulation
    :param nitermax: int, optional
        Maximum number of iterations
    :param tol: float, optional
        The iteration stops when |P - P_old| < tol
    :return:
        The transport plan w.r.t. the value of reg
    """

    # define objective function f
    def f(G):
        G = G.reshape((a.shape[0], b.shape[0]))
        return np.sum(G * C) + reg * KL_divergence(G.sum(1), a) + reg * KL_divergence(G.sum(0), b)

    # define gradient of f
    def df(G):
        G = G.reshape((a.shape[0], b.shape[0]))
        return (C + reg * np.outer(np.log(G.sum(1)/a + 1e-16), np.ones(b.shape[0])) + reg * np.outer(np.ones(a.shape[0]), np.log(G.sum(0) / b+1e-16))).ravel()

    G0 = np.ones(a.shape[0] * b.shape[0])
    bounds = sp.optimize.Bounds(np.zeros(a.shape[0] * b.shape[0]), np.inf * np.ones(a.shape[0] * b.shape[0]), keep_feasible=False)
    res = sp.optimize.minimize(f, G0, jac=df, method='L-BFGS-B', bounds=bounds, options={'ftol': tol, 'gtol': 1e-16, 'maxiter': nitermax})
    x = res.x.reshape((a.shape[0], b.shape[0]))
    return x


def ot_uklreg_solve_mm(C, a, b, reg, nitermax=100000, tol=1e-14, P0=None,  verbose=False):
    """ Majorization-Minimization algorithm for entropy-regularized and KL-penalized UOT (same regularization coefficient)

    :param C: array, shape(n,m)
        Transport cost
    :param a: array, shape(n,)
        Histogram of source distribution
    :param b: array, shape(m,)
        Histogram of target distribution
    :param reg: float
        Penalization coefficient in UOT formulation
    :param nitermax: int, optional
        Maximum number of iterations
    :param tol: float, optional
        The iteration stops when |P - P_old| < tol
    :param P0: array, shape(n,m), optional
        Initialization of transport plan P
    :param verbose: bool, optional
        If verbose is true, print the error at each iteration
    :return:
        The transport plan w.r.t. the value of reg
    """

    if P0 is None:
        P = a[:, None] * b[None, :]
    else:
        P = P0

    for i in range(nitermax):
        Pold = P.copy()
        P = np.exp((2 * np.log(P) - C / reg + np.log(a / (P.sum(1) + 1e-16))[:, None] + np.log(
            (b / (P.sum(0) + 1e-16)))[None, :]) / 3)
        if verbose:
            print(np.linalg.norm(P - Pold))
        if np.linalg.norm(P - Pold) < tol:
            break

    return P
