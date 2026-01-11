import numpy as np

def sinkhorn_knopp(
    K,
    n_iters=50,
    r=None,
    c=None,
    eps=1e-12,
    return_scalings=False,
    check_finite=True,
):
    """
    Sinkhorn–Knopp matrix balancing.

    Finds u, v >= 0 such that P = diag(u) @ K @ diag(v) has row sums r and column sums c.
    Special case r=c=ones => approximately doubly stochastic.

    Parameters
    ----------
    K : (n, m) array_like
        Nonnegative matrix. For best behavior use strictly positive entries.
    n_iters : int
        Number of iterations.
    r : (n,) array_like or None
        Target row sums. If None => uniform (sum=1): r = ones(n)/n.
    c : (m,) array_like or None
        Target column sums. If None => uniform (sum=1): c = ones(m)/m.
    eps : float
        Small constant to avoid division by zero.
    return_scalings : bool
        If True, also return u, v.
    check_finite : bool
        If True, error if K has NaN/Inf.

    Returns
    -------
    P : (n, m) ndarray
        Balanced matrix.
    (optional) u : (n,) ndarray, v : (m,) ndarray
        Scaling vectors.
    """
    K = np.asarray(K, dtype=np.float64)

    if K.ndim != 2:
        raise ValueError("K must be 2D.")
    n, m = K.shape

    if check_finite and (not np.isfinite(K).all()):
        raise ValueError("K contains NaN or Inf.")

    if np.any(K < 0):
        raise ValueError("K must be nonnegative.")

    # Default targets: uniform distributions (sum to 1)
    if r is None:
        r = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        if r.shape[0] != n:
            raise ValueError("r has wrong shape.")
        if np.any(r < 0):
            raise ValueError("r must be nonnegative.")
        s = r.sum()
        if s <= 0:
            raise ValueError("r must have positive sum.")
        r = r / s

    if c is None:
        c = np.full(m, 1.0 / m, dtype=np.float64)
    else:
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        if c.shape[0] != m:
            raise ValueError("c has wrong shape.")
        if np.any(c < 0):
            raise ValueError("c must be nonnegative.")
        s = c.sum()
        if s <= 0:
            raise ValueError("c must have positive sum.")
        c = c / s

    # If K has zero rows/cols, updates can divide by zero. eps helps but may not "fix" feasibility.
    u = np.ones(n, dtype=np.float64)
    v = np.ones(m, dtype=np.float64)

    # Main iterations
    for _ in range(int(n_iters)):
        Kv = K @ v
        u = r / np.maximum(Kv, eps)

        KTu = K.T @ u
        v = c / np.maximum(KTu, eps)

    P = (u[:, None] * K) * v[None, :]

    if return_scalings:
        return P, u, v
    return P


def is_doubly_stochastic(A, tol=1e-6):
    """
    Quick check for (approximately) doubly stochastic square matrix:
    nonnegative, rows sum to 1, cols sum to 1.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    if np.any(A < -tol):
        return False
    row_ok = np.allclose(A.sum(axis=1), 1.0, atol=tol, rtol=0)
    col_ok = np.allclose(A.sum(axis=0), 1.0, atol=tol, rtol=0)
    return row_ok and col_ok


if __name__ == "__main__":
    # Example: make a random matrix doubly stochastic
    rng = np.random.default_rng(0)
    S = rng.normal(size=(4, 4))

    # Common ML trick: make K strictly positive
    K = np.exp(S)

    P = sinkhorn_knopp(K, n_iters=50)  # targets default to uniform row/col sums

    print("Row sums:", P.sum(axis=1))
    print("Col sums:", P.sum(axis=0))
    print("Is doubly stochastic (approx)?", is_doubly_stochastic(P, tol=1e-6))
