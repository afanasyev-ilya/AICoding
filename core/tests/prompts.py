MATMUL_PROMPT = """import numpy as np


def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    '''
    Compute C = A @ B using explicit Python loops.
    A: shape (n, k)
    B: shape (k, m)
    Returns:
        C: shape (n, m)
    '''
    assert A.ndim == 2 and B.ndim == 2
    n, k = A.shape
    k2, m = B.shape
    assert k == k2

    C = np.zeros((n, m), dtype=A.dtype)

    for i in range(n):
        for j in range(m):
            s = 0
            for """
