import numpy as np
import scipy.sparse as sp
import scipy
import ot
import celer
from sklearn.linear_model import Lasso


def complement_schur(idx_i, idx_j, H_inv_current, id_pop):
    # Update the matrix (H^T*H)^-1 by Schur complement

    n = len(idx_i)
    if id_pop != -1:
        b = H_inv_current[id_pop,:]
        b = np.delete(b, id_pop)
        H_inv_del = np.delete(H_inv_current, id_pop, 0)
        a = H_inv_del[:, id_pop]
        H_inv_del = np.delete(H_inv_del, id_pop, 1)
        H_inv = H_inv_del - np.outer(a, b)/H_inv_current[id_pop, id_pop]
    else:
        b = np.zeros((n-1,1))
        b[:, 0] = (np.array(idx_i)[:-1] == idx_i[-1]) * 1. + (np.array(idx_j)[:-1] == idx_j[-1]) * 1.
        if np.shape(b)[0] == 0:
            H_inv = np.array([[1/2]])
        else:
            X = H_inv_current.dot(b)
            s = 2 - b.T.dot(X)
            H_inv = np.zeros((n, n))
            H_inv[:-1, :-1] = H_inv_current + X.dot(X.T)/s
            H_inv[-1, :-1] = -X.ravel()/s
            H_inv[:-1, -1] = -X.ravel()/s
            H_inv[-1, -1] = 1/s
    return H_inv


def compute_lambda_a(idx_i, idx_j, pt, delta, C, a, b, current_lk, ones_m, ones_n):
    # compute the value of lambda when one component enters active set

    dim = np.shape(C)
    if dim[0] + dim[1] > 600:
        # sparse setup
        Pts = sp.coo_matrix((pt, (idx_i, idx_j)), shape=dim)
        Deltas = sp.coo_matrix((delta, (idx_i, idx_j)), shape=dim)
        M = -(C + 2 * (Deltas.dot(ones_m)[:, None] + Deltas.T.dot(ones_n)[None, :])) / ((Pts.dot(ones_m) - a)[:, None] + (Pts.T.dot(ones_n) - b)[None, :]+1e-16) / 2
    else:
        # non sparse setup
        Pt = np.zeros(dim)
        Delta = np.zeros(dim)
        Pt[idx_i, idx_j] = pt
        Delta[idx_i, idx_j] = delta
        M = -(C + 2 * np.sum(Delta, 1)[:,None] + 2 * np.sum(Delta, 0)[None,:]) / ((np.sum(Pt, 1) - a)[:, None] + (np.sum(Pt, 0) - b)[None, :]+1e-16) / 2
    M[idx_i, idx_j] = np.inf
    M[M <= (current_lk+ 1e-8*current_lk)] = np.inf
    return M


def compute_lambda_r(delta, pt, current_lk):
    # compute the value of lambda when one component leaves active set

    if len(delta) == 1:
        return np.inf, 0
    l = -delta / (pt + 1e-16)
    l[l <= (current_lk + 1e-8*current_lk)] = np.inf
    if np.all(l) == np.inf:
        return np.inf, 0
    else:
        return np.min(l), np.argmin(l)


def compute_transport_plan(lam, lambda_list, Pi_list):
    """ Compute the transport plan P in regularization path for any given value of lambda

    :param lam: float
        Value of regularization coefficient lambda
    :param lambda_list: list
        List of all lambdas in regularization path
    :param Pi_list:
        List of all transport plans in regularization path
    :return:
        Transport plan corresponding to given lambda
    """

    if lam <= lambda_list[0]:
        Pi_inter = np.zeros(np.shape(Pi_list[-1]))
    elif lam >= lambda_list[-1]:
        Pi_inter = Pi_list[-1].toarray()
    else:
        idx = np.where(lambda_list < lam)[0][-1]
        lam_k = lambda_list[idx]
        lam_k1 = lambda_list[idx+1]
        pi_k = Pi_list[idx]
        pi_k1 = Pi_list[idx+1]
        Pi_inter = pi_k + (pi_k1-pi_k)*(1/lam - 1/lam_k) / (1/lam_k1 - 1/lam_k)
        Pi_inter = Pi_inter.toarray()
    return Pi_inter


def ot_ul2_reg_path(a: np.array, b: np.array, C: np.array, lambdamax=np.inf, savePi=False, itmax=50000, save_AT_length=False):
    """ Function of regularized path for l2-panalized UOT

    :param a: array, shape (n,)
        Histogram of source distribution
    :param b: array, shape (m,)
        Histogram of target distribution
    :param C: array, shape (n, m)
        Cost matrix
    :param lambdamax: float, optional
        Maxmium value of regularization coefficient lambda (The default value of lambda is inf)
    :param savePi: bool, optional
        If savePi is true, transport plans in regularization path are stored
    :param itmax: int, optional
        Maximum number of iterations

    :return:
        Transport plan w.r.t. the value of lambda
        Final value of lambda
        List of transport plan in regularization path (If savePi is true)
        List of lambda in regularized path
        Total iteration number
    """

    n = np.shape(a)[0]
    m = np.shape(b)[0]
    ones_n = np.ones((n,))
    ones_m = np.ones((m,))

    n_iter = 0
    lambda_list = []
    Pi_list = []

    active_index_i = []
    active_index_j = []
    e = np.array([])
    c = np.array([])
    H_inv = np.array([[]])
    lam = 0

    active_set_length = []

    while n_iter < itmax:
        # deal with the first iteration
        # print('------------iteration ', n_iter, '--------------')
        # print('active set length:', len(active_index_i))
        active_set_length.append(len(active_index_i))
        if n_iter == 0:
            M = C/(a[:, None] + b[None, :])/2
            ik, jk = np.unravel_index(np.argmin(M), M.shape)
            lam = M[ik, jk]
            id_pop = -1
            delta = np.array([])
            pi_tilde = np.array([])
        else:
            # compute next lambda when a couple of index is added to the active set
            M = compute_lambda_a(active_index_i, active_index_j, pi_tilde, delta, C, a, b, lam, ones_m, ones_n)

            # compute the next lambda when a couple of index is removed from the active set
            alt_lam, id_pop = compute_lambda_r(delta, pi_tilde, lam)
            lam = np.min(M)

            if alt_lam < lam:
                lam = alt_lam
            else:
                ik, jk = np.unravel_index(np.argmin(M), M.shape)
                id_pop = -1

        if lambdamax == np.inf:
            # stop criteria on marginals
            if n_iter > 0:
                pi_vect = delta / lam + pi_tilde
                Pi = sp.coo_matrix((pi_vect, (active_index_i, active_index_j)), shape=(n, m))
                if np.linalg.norm(Pi.dot(ones_m)-a, ord=2) + np.linalg.norm(Pi.T.dot(ones_n)-b, ord=2) <1e-6:
                    if savePi:
                        Pi_list.append(Pi)
                    lambda_list.append(lam)
                    break
        else:
            # stop criteria on lambda
            if lam > lambdamax:
                pi_vect = delta / lambdamax + pi_tilde
                Pi= sp.coo_matrix((pi_vect, (active_index_i, active_index_j)), shape=(n, m))
                if savePi:
                    Pi_list.append(Pi)
                lambda_list.append(lam)
                break

        # if the positivity constraint is not satisfied, remove index (i,j) from the current active set
        # otherwise add (ik,jk) found from M to active set
        if id_pop != -1:
            active_index_j.pop(id_pop)
            active_index_i.pop(id_pop)
            c = np.delete(c, id_pop, 0)
            e = np.delete(e, id_pop, 0)

        else:
            active_index_i.append(ik)
            active_index_j.append(jk)
            c = np.append(c, -C[ik, jk] / 2)
            e = np.append(e, a[ik] + b[jk])


        # compute H^-1 (Schur complement)
        H_inv = complement_schur(active_index_i, active_index_j, H_inv, id_pop)
        delta = H_inv @ c
        pi_tilde = H_inv @ e
        pi_vect = delta / lam + pi_tilde

        # Compute current transport plan Pi
        if savePi:
            Pi = sp.coo_matrix((pi_vect, (active_index_i, active_index_j)), shape=(n, m))
            Pi_list.append(Pi)

        lambda_list.append(lam)
        n_iter += 1

    if itmax <= n_iter:
        Pi = sp.coo_matrix((pi_vect, (active_index_i, active_index_j)), shape=(n, m))
        print('max iteration number reached')
    if savePi:
        if save_AT_length:
            return Pi_list[-1].toarray(), lam, Pi_list, np.array(lambda_list), n_iter, active_set_length
        else:
            return Pi_list[-1].toarray(), lam, Pi_list, np.array(lambda_list), n_iter
    else:
        if save_AT_length:
            return Pi.toarray(), lam, np.array(lambda_list), n_iter, active_set_length
        else:
            return Pi.toarray(), lam, np.array(lambda_list), n_iter


def ot_ul2_solve_BFGS(C, a, b, reg, maxiter=100000, tol=1e-14):
    """  BFGS algorithm for l2-penalized UOT

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
        Transport plan w.r.t. the value of reg
    """

    # define objective function f
    def f(G):
        G = G.reshape((a.shape[0], b.shape[0]))
        return np.sum(G * C) + reg * np.sum((G.sum(1) - a) ** 2) + reg * np.sum((G.sum(0) - b) ** 2)

    # define the gradient of f
    def df(G):
        G = G.reshape((a.shape[0], b.shape[0]))
        return (C + reg * 2 * np.outer((G.sum(1) - a), np.ones(b.shape[0])) + reg * 2 * np.outer(np.ones(a.shape[0]),G.sum(0) - b)).ravel()

    G0 = np.zeros(a.shape[0] * b.shape[0])
    bounds = scipy.optimize.Bounds(np.zeros(a.shape[0] * b.shape[0]), np.inf * np.ones(a.shape[0] * b.shape[0]), keep_feasible=False)
    res = scipy.optimize.minimize(f, G0, jac=df, method='L-BFGS-B', bounds=bounds, options={'ftol': tol, 'gtol': 1e-16, 'maxiter': maxiter})
    x = res.x.reshape((a.shape[0], b.shape[0]))
    return x


def get_X_lasso(n, m):
    """ Construction of design matrix H

    :param n: int
        Dimension of source histogram
    :param m: int
        Dimension of target histogram
    :return:
        Design (or dictionary) matrix H
    """

    jHa = np.arange(m * n)
    iHa = np.repeat(np.arange(n), m)
    jHb = np.arange(m * n)
    iHb = np.tile(np.arange(m), n) + n
    j = np.concatenate((jHa, jHb))
    i = np.concatenate((iHa, iHb))
    H = sp.csc_matrix((np.ones(n * m * 2), (i, j)), shape=(n+m, n*m))
    return H


def ot_ul2_solve_lasso_celer(C, a, b, reg, nitermax=100000, tol=1e-14):
    """ Celer algorithm for lasso-formulated l2-penalized UOT

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
    :return:
        The transport plan w.r.t. the value of reg
    """

    X = get_X_lasso(C.shape[0], C.shape[1])
    y = np.concatenate((a, b))
    reg2 = 1.0 / (2 * (C.shape[0] + C.shape[1]) * reg)
    model = celer.Lasso(reg2, max_iter=nitermax, weights=C.ravel(), positive=True, fit_intercept=False, tol=tol)
    model.fit(X, y)
    G2 = model.coef_.reshape(C.shape)
    return G2


def ot_ul2_solve_lasso_cd(C, a, b, reg, nitermax=100000, tol=1e-14):
    """ Coordinate descent algorithm for lasso-formulated l2-penalized UOT

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
    :return:
        The transport plan w.r.t. the value of reg
    """

    X = get_X_lasso(C.shape[0], C.shape[1])
    X = X.dot(sp.diags((1 / C.ravel())))
    y = np.concatenate((a, b))
    reg2 = 1.0 / (2 * (C.shape[0] + C.shape[1]) * reg)
    model = Lasso(reg2, positive=True, fit_intercept=False, max_iter=nitermax, tol=tol)
    model.fit(X, y)
    G2 = model.coef_.reshape(C.shape) / C
    return G2


def ot_ul2_solve_mu(C, a, b, reg, nitermax=100000, tol=1e-14, P0=None, verbose=False):
    """ Majorization-Minimization algorithm for l2-penalized UOT

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
    abt = np.maximum(a[:, None] + b[None, :] - C / (2 * reg), 0)
    for i in range(nitermax):
        Pold = P.copy()
        P = P * abt / (P.sum(0, keepdims=True) + P.sum(1, keepdims=True) + 1e-16)
        pmax = P.max()
        P = P * (P > pmax * 1e-16)
        if verbose:
            print(np.linalg.norm(P - Pold))
        if np.linalg.norm(P - Pold) < tol:
            break
    return P