"""gnmf.py"""
import sys
import numpy as np
import numpy.linalg as LA
from scipy.sparse import coo_matrix
from sklearn.metrics import pairwise_distances


def objective(X, U, V, L):
    """Compute the value of objective function for GNMF

    Parameters
    ----------
    X: given matrix
    U: factor matrix (basis)
    V: factor matrix (coefficient)
    L: graph Laplacian

    Return
    -------
    obj_val: objective function value
    """
    obj_val = LA.norm(X - np.dot(U, V.T), "fro") ** 2 + np.trace(np.dot(V.T, np.dot(L, V)))
    return obj_val


def initializeUV(row, col, dim, random_state=None):
    """Initialize factor matrices

    Parameters
    ----------
    row: row size of given matrix
    col: column size of given matrix
    dim: given dimension size
    random_state: seed value

    Returns
    -------
    U: factor matrix (basis)
    V: factor matrix (coefficient)
    """
    np.random.seed(random_state)
    U = np.random.rand(row * dim).reshape(row, dim)
    V = np.random.rand(col * dim).reshape(col, dim)
    return U, V


def preproc_ncw(X):
    """normalized cut weighting preprocessing:
    W. Xu, X. Liu, and Y. Gong, “Document clustering based on non-negative matrix factorization,”
    Proc. Ann. Int’l ACM SIGIR Conf. Research and Development in Information Retrieval, pp. 267-273, 2003.

    Parameter
    ----------
    X: given matrix

    Return
    -------
    Xncw: normalized cut weighted matrix X
    """
    dmat = np.diag(np.dot(np.dot(X.T, X), np.ones(X.shape[1])))
    for i in range(X.shape[1]):
        dmat[i, i] = np.sqrt(1.0 / dmat[i, i])
    Xncw = np.dot(X, dmat)
    return Xncw


def const_pNNgraph(X, n_neighbors=5):
    """construct p-nearest neighbor graph

    Parameters
    ----------
    X: given matrix
    n_neighbors: p value
    wtype:
        0: 0-1 weighting
        1: heat kernel weighting
        2: dot product weighting
    sigma: used if wtype is 1

    Return
    -------
    W: adjacency matrix of p-NN graph
    """
    m, n = X.shape
    maxv = sys.float_info.max
    XT_csr = coo_matrix(X.T).tocsr()
    distance_mat = pairwise_distances(X=XT_csr, Y=XT_csr, metric="euclidean", n_jobs=-1)
    distance_mat += maxv * np.identity(n)
    idx = np.argsort(distance_mat)[:, :n_neighbors]
    distance_mat -= maxv * np.identity(n)
    W = np.zeros((n, n))
    for i in range(n):
        W[i, idx[i]] = 1.0
    W = W + W.T
    W = np.where(W > 0.0, 1.0, 0.0)

    return W


def normalizeUV(U, V):
    """delete ambiguous expression for U and V by using normalization

    Parameters
    ----------
    U: factor matrix (basis)
    V: factor matrix (coefficient)

    Returns
    -------
    U: unique factor matrix (basis)
    V: unique factor matrix (coefficient)
    """
    Unrms = LA.norm(U, axis=0)
    Uuni = U / Unrms
    Vuni = V * Unrms
    return Uuni, Vuni


def mur_algorithm(X, U, V, W, D):
    """Modified multiplicative update rules algorithm

    Parameters
    ----------
    X: given matrix
    U: factor matrix (basis)
    V: factor matrix (coefficient)
    W: adjacency matrix of p-NN graph
    D: degree matrix of W

    Returns
    -------
    U: updated factor matrix (basis)
    V: updated factor matrix (coefficient)
    """
    Unew = U * np.dot(X, V) / np.dot(U, np.dot(V.T, V))
    Unew = np.fmax(Unew, 1e-9)
    Vnew = V * (np.dot(X.T, Unew) + np.dot(W, V)) / (np.dot(V, np.dot(Unew.T, Unew)) + np.dot(D, V))
    Vnew = np.fmax(Vnew, 1e-9)

    return Unew, Vnew


class GNMF:
    """
    Conduct graph regularized nonnegative matrix factorization (GNMF).
    """

    def __init__(
        self,
        n_components,
        rterm=100.0,
        p=5,
        max_iter=100,
        W=None,
        ncw=True,
        random_state=None,
        calc_objs=None,
        verbose=False,
    ):
        if not (n_components > 0):
            raise (ValueError("n_components must be positive."))
        if not (rterm >= 0.0):
            raise (ValueError("rterm must be positive."))
        if not (p > 0):
            raise (ValueError("p must be positive."))
        if not (max_iter > 1):
            raise (ValueError("maxiter must be greater than 1."))

        self.n_components = n_components
        self.rterm = rterm
        self.p = p
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.W = W
        self.ncw = ncw
        self.calc_objs = calc_objs

    def fit(self, _X):
        if self.ncw:
            X = preproc_ncw(_X.T)
        else:
            X = np.copy(_X.T)
        fea, smp = X.shape

        # init
        U, V = initializeUV(fea, smp, self.n_components, self.random_state)
        U = np.fmax(U, 1e-9)
        V = np.fmax(V, 1e-9)
        U, V = normalizeUV(U, V)

        # construct p-NN graph
        if self.W is None:
            W = const_pNNgraph(_X.T, self.p)
        else:
            W = np.copy(self.W)
        W = self.rterm * W
        D = np.diag(np.sum(W, axis=1))

        if self.calc_objs:
            _L = D - W
            dmat = np.dot(X.T, np.dot(X, np.ones(smp)))
            dmat_half = np.sqrt(dmat)
            L = np.dot(dmat_half, np.dot(_L, dmat_half))
            init = objective(X, U, V, L)
            objs = [1.0]

        # MUR
        for it in range(1, self.max_iter + 1):
            U, V = mur_algorithm(X, U, V, W, D)

            if self.calc_objs:
                obj_val = objective(X, U, V, L)
                objs.append(obj_val / init)

        self.U, self.V = normalizeUV(U, V)

        if self.calc_objs:
            self.objs = objs

        return self

    def get_basis(self):
        """getter for basis"""
        return self.U

    def get_coef(self):
        """getter for coefficient"""
        return self.V

    def get_objs(self):
        """getter for objective function values per iteration"""
        if not self.calc_objs:
            print("objective function values: not calculated")
            return
        return self.objs
