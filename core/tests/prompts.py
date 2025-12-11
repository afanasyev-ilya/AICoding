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
"""

BINARY_SEARCH_PROMPT = """import numpy as np

def binary_search(arr, target):
    '''
    Performs a binary search on a sorted list to find a target element.

    Args:
        arr: The sorted list to search within.
        target: The element to search for.

    Returns:
        The index of the target element if found, otherwise -1.
    '''
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2  # Calculate the middle index
        mid_val = arr[mid]       # Get the value at the middle index

"""