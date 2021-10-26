import numpy as np
import scipy.sparse as sp

def complement_schur(idx_i, idx_j, H_inv_current, id_pop, m):
    # Update the matrix (H^T*H)^-1 by Schur complement

    n = len(idx_i)
    if id_pop != -1:
        b = H_inv_current[id_pop+m,:]
        b = np.delete(b, id_pop+m)
        H_inv_del = np.delete(H_inv_current, id_pop+m, 0)
        a = H_inv_del[:, id_pop+m]
        H_inv_del = np.delete(H_inv_del, id_pop+m, 1)
        H_inv = H_inv_del - np.outer(a, b)/H_inv_current[id_pop+m, id_pop+m]
    else:
        b = np.zeros((m+n-1,1))
        b[idx_j[-1], 0] = 1.
        b[m:, 0] = (np.array(idx_i)[:-1] == idx_i[-1]) * 1.
        X = H_inv_current.dot(b)
        s = 1 - b.T.dot(X)
        H_inv = np.zeros((m+n, m+n))
        H_inv[:-1, :-1] = H_inv_current + X.dot(X.T)/s
        H_inv[-1, :-1] = -X.ravel()/s
        H_inv[:-1, -1] = -X.ravel()/s
        H_inv[-1, -1] = 1/s
    return H_inv


def compute_lambda_a(idx_i, idx_j, alpha, beta, C, a, current_lk, o_m):
    # compute the value of lambda when one component enters active set

    n = np.shape(C)[0]
    m = np.shape(C)[1]
    delta_u = alpha[0:m]
    ut = beta[0:m]
    if n + m > 600:
        # sparse setup
        Pts = sp.coo_matrix((beta[m:], (idx_i, idx_j)), shape=(n, m))
        Deltas = sp.coo_matrix((alpha[m:], (idx_i, idx_j)), shape=(n, m))
        M = -(C + Deltas.dot(o_m)[:, None] + delta_u[None, :]) / ((Pts.dot(o_m) - a)[:, None] + ut[None, :]+1e-16)
    else:
        # non sparse setup
        Pt = np.zeros((n, m))
        Delta_pi = np.zeros((n, m))
        Pt[idx_i, idx_j] = beta[m:]
        Delta_pi[idx_i, idx_j] = alpha[m:]
        M = -(C + np.sum(Delta_pi, 1)[:, None] + delta_u[None, :]) / ((np.sum(Pt, 1) - a)[:, None] + ut[None, :]+1e-16)

    M[idx_i, idx_j] = np.inf
    M[M <= (current_lk+ 1e-8*current_lk)] = np.inf
    return M


def compute_lambda_r(alpha, beta, current_lk, m):
    # compute the value of lambda when one component leaves active set

    delta = alpha[m:]
    pt = beta[m:]
    l = -delta / (pt + 1e-16)
    l[pt >= 0] = np.inf
    l[l <= current_lk] = np.inf
    if np.all(l) == np.inf:
        return np.inf, 0
    else:
        return np.min(l), np.argmin(l)


# useless if the Schur complement is applied
def construct_H(idx_i, idx_j, m):
    k = len(idx_i)
    H = np.zeros((k+m, k+m))
    for i in range(k):
        # construct K_A
        for j in range(i):
            if idx_i[i] == idx_i[j]:
                H[m+i, m+j] += 1
                H[m+j, m+i] += 1
        H[i+m, i+m] = 1
        # construct G_A
        H[m+i, idx_j[i]] = 1
    # construct F_A
    for i in range(m):
        for j in range(k):
            if idx_j[j] == i:
                H[i, m+j] = 1
    return H


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


def ot_semi_relaxed_ul2_reg_path(a: np.array, b: np.array, C: np.array, lambdamax=np.inf, itmax=50000):
    """ Function of regularized path for l2-panalized UOT

    :param a: array, shape (n,)
        Histogram of source distribution
    :param b: array, shape (m,)
        Histogram of target distribution
    :param C: array, shape (n, m)
        Cost matrix
    :param lambdamax: float, optional
        Maxmium value of regularization coefficient lambda (The default value of lambda is inf)
    :param itmax: int, optional
        Maximum number of iterations

    :return:
        Transport plan w.r.t. the value of lambda
        Final value of lambda
        List of transport plan in regularization path
        List of lambda in regularized path
        Total iteration number
    """

    n = np.shape(a)[0]
    m = np.shape(b)[0]
    ones_m = np.ones((m,))
    n_iter = 0

    active_index_i = []
    active_index_j = []
    e = b
    c = np.zeros((m,))

    # initialization
    for j in range(np.shape(C)[1]):
        i = np.argmin(C[:, j])
        active_index_i.append(i)
        active_index_j.append(j)
        c = np.append(c, -C[i, j])
        e = np.append(e, a[i])
    H = construct_H(active_index_i, active_index_j, m)
    H_inv = np.linalg.inv(H)
    alpha = np.linalg.solve(H, c)
    beta = np.linalg.solve(H, e)

    lambda_list = [0]
    Pi_list = []
    Pi0 = sp.coo_matrix((b, (active_index_i, active_index_j)), shape=(n, m))
    Pi_list.append(Pi0)
    lam = 0

    while n_iter < itmax:
        # compute next lambda when a point is added to the active set
        M = compute_lambda_a(active_index_i, active_index_j, alpha, beta, C, a, lam, ones_m)

        # compute the next lambda a point is removed from the active set
        alt_lam, id_pop = compute_lambda_r(alpha, beta, lam, m)
        lam = np.min(M)

        if alt_lam < lam:
            lam = alt_lam
        else:
            ik, jk = np.unravel_index(np.argmin(M), M.shape)
            id_pop = -1

        if lambdamax == np.inf:
            # stop criteria on the marginal
            delta = alpha[m:]
            pi_tilde = beta[m:]
            pi_vect = delta / lam + pi_tilde
            Pi = sp.coo_matrix((pi_vect, (active_index_i, active_index_j)), shape=(n, m))
            if np.linalg.norm(Pi.dot(ones_m) - a, ord=2) < 1e-4:
                Pi_list.append(Pi)
                lambda_list.append(lam)
                break
        else:
            # stop criteria on lambda
            if lam > lambdamax:
                delta = alpha[m:]
                pi_tilde = beta[m:]
                pi_vect = delta / lambdamax + pi_tilde
                Pi= sp.coo_matrix((pi_vect, (active_index_i, active_index_j)), shape=(n, m))
                Pi_list.append(Pi)
                lambda_list.append(lam)
                break

        # if the positivity constraint is not satisfied, remove index (i,j) from current active set,
        # otherwise, add (ik jk) found from M to active set
        if id_pop != -1:
            active_index_j.pop(id_pop)
            active_index_i.pop(id_pop)
            c = np.delete(c, id_pop + m, 0)
            e = np.delete(e, id_pop + m, 0)
        else:
            active_index_i.append(ik)
            active_index_j.append(jk)
            c = np.append(c, -C[ik, jk])
            e = np.append(e, a[ik])

        # Compute H_inv (Schur complement)
        H_inv = complement_schur(active_index_i, active_index_j, H_inv, id_pop, m)
        alpha = H_inv @ c
        beta = H_inv @ e
        pi_vect = alpha[m:] / lam + beta[m:]

        # Compute current transport plan
        Pi = sp.coo_matrix((pi_vect, (active_index_i, active_index_j)), shape=(n, m))
        lambda_list.append(lam)
        Pi_list.append(Pi)
        n_iter += 1

    if itmax <= n_iter:
        print('maximal iteration number reached')

    return Pi_list[-1].toarray(), lam, Pi_list, np.array(lambda_list), n_iter

