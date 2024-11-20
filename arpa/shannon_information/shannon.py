import numpy as np
import warnings
from multiprocess import Pool, cpu_count
import tqdm
from arpa.shannon_information.mutual_info import compute_mi, compute_cmi
from typing import Tuple,Callable

from arpa.utils.symmetric_operators import make_integer


def cov(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the covariance between two arrays."""
    return ((a - np.mean(a)) * (b - np.mean(b))).sum() / (len(a) - 1)

def norm_cov(a: np.ndarray, b: np.ndarray, epison=10**-9) -> float:
    """Calculate the normalized covariance (correlation coefficient) between two arrays."""
    covariance = cov(a, b)
    a_std = np.std(a)
    b_std = np.std(b)
    if a_std == 0:
        a_std = epison
    if b_std == 0:
        b_std = epison
    normalized_covariance = covariance / (a_std * b_std)
    return normalized_covariance

def process_call_cmi(*args: Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[Tuple[int, int], float]:
    """Process call for conditional mutual information."""
    i, j, x, y, target = args[0]
    if i == j:
        costs = compute_mi(x, target)
    else:
        costs = compute_cmi(x, y, target)
    return (i, j), costs

def process_call_icmi(*args: Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[Tuple[int, int], float]:
    """Process call for incremental conditional mutual information."""
    i, j, x, y, target = args[0]
    if i == j:
        costs = compute_mi(x, target)
    else:
        costs = np.max([0, compute_cmi(y, target, x)])
    return (i, j), costs

def process_call_mi(*args: Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[Tuple[int, int], float]:
    """Process call for mutual information."""
    i, j, x, y, target = args[0]
    if i == j:
        costs = compute_mi(x, target)
    else:
        costs = compute_mi(x, y)
    return (i, j), costs

def process_call_cov(*args: Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[Tuple[int, int], float]:
    """Process call for covariance."""
    i, j, x, y, target = args[0]
    if i == j:
        costs = np.abs(cov(x, target))
    else:
        costs = np.abs(cov(x, y))
    if np.isnan(costs):
        costs = 0
    return (i, j), costs

def process_call_corrcoef(*args: Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[Tuple[int, int], float]:
    """Process call for correlation coefficient."""
    i, j, x, y, target = args[0]
    if i == j:
        costs = np.abs(norm_cov(x, target))
    else:
        costs = np.abs(norm_cov(x, y))
    if np.isnan(costs):
        costs = 0
    return (i, j), costs

def process_call_mi_cov(*args: Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[Tuple[int, int], float]:
    """Process call for mutual information and covariance."""
    i, j, x, y, target = args[0]
    if i == j:
        costs = compute_mi(x, target)
    else:
        costs = np.abs(np.corrcoef(x, y)[0][1])
    if np.isnan(costs):
        costs = 0
    return (i, j), costs

def calculate_information_matrix(matrix: np.ndarray, target: np.ndarray, verbose: int, is_symmetric: bool,
                                 discretize: bool, func: Callable = process_call_mi, precision:int = 100000,
                                 processes: int = 1) -> np.ndarray:
    """Calculate the information matrix using the specified function.

    :param matrix: Input data matrix.
    :param target: Target variable.
    :param verbose: Verbosity level.
    :param is_symmetric: Whether the matrix is symmetric.
    :param discretize: Whether to discretize the data.
    :param func: Function to use for calculation.
    :param processes: Number of processes to use.
    :returns: Information matrix.
    """
    size = matrix.shape[1]
    costs = np.zeros((size, size))
    values = []

    if target.dtype.char not in np.typecodes['AllInteger'] and discretize:
        warnings.warn(f"Target variable is not integer. Setting to Integer."
                      f"The target will be multiplied by {precision}")
        target = make_integer(target, precision)

    if matrix.dtype.char not in np.typecodes['AllInteger'] and discretize:
        warnings.warn(f"Cost Matrix variable is not integer. Setting to Integer."
                      f"The matrix will be multiplied by {precision}")
        target = make_integer(matrix, precision)

    if func.__name__ == "process_call_mi_cov":
        costs = np.abs(np.corrcoef(matrix.T))
        for i in range(size):
            X = matrix[:, i]
            values.append((i, i, X, X, target))
    elif func.__name__ == "process_call_corrcoef":
        costs = np.abs(np.corrcoef(matrix.T))
        for i in range(size):
            X = matrix[:, i]
            values.append((i, i, X, X, target))
    elif func.__name__ == "process_call_cov":
        costs = np.abs(np.cov(matrix.T))
        for i in range(size):
            X = matrix[:, i]
            values.append((i, i, X, X, target))
    else:
        for i in range(size):
            X = matrix[:, i]
            if not is_symmetric:
                for j in range(size):
                    Y = matrix[:, j]
                    values.append((i, j, X, Y, target))
            else:
                for j in range(i, size):
                    Y = matrix[:, j]
                    values.append((i, j, X, Y, target))

    if verbose > 0:
        total = len(values)
        pool = Pool(processes=processes)
        results = list(tqdm.tqdm(pool.imap_unordered(func, iter(values)), total=total))
    else:
        with Pool(processes=processes) as pool:
            results = list(pool.imap_unordered(func, iter(values)))

    for t in results:
        costs[t[0]] = t[1]

    if is_symmetric:
        for t in results:
            costs[t[0][::-1]] = t[1]
    np.nan_to_num(costs, 0)
    return costs