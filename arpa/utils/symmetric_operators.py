import numpy as np
import pandas as pd


def fill_diagonal(matrix, value):
    for i in range(len(matrix)):
        matrix[i][i] = value
    return matrix

def make_integer(maxtix, precision):
    return np.ceil(maxtix * precision).astype(int)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    if type(list):
        a = np.array(a)
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def check_indexes_symmetry(indices):
    return all((j, i) in indices for i, j in indices)


# def make_symmetric(matrix):
#     return (matrix + matrix.T) / 2

def make_symmetric(matrix):
    if type(matrix) == list:
        matrix = np.array(matrix)
    matrix = (matrix + matrix.T) / 2
    return matrix


def add_symmetric_noise(matrix, noise=0.0001):
    symmetric_noise = make_symmetric(np.random.rand(*matrix.shape) * noise)
    symmetric_noise = np.diag(symmetric_noise.diagonal())
    return matrix + symmetric_noise

def force_symetric_assigment(matrix, costs):
    # get size of the costs matrix
    size = costs.shape[0]
    find_non_symetric = (matrix + matrix.T) / 2
    # identify where the matrix is not symmetric
    (rows_one, cols_one) = np.where(find_non_symetric == 1)
    # identify where the matrix is symmetric
    (rows, cols) = np.where(find_non_symetric == 0.5)
    # the rows will be duplicated where there is no symetry, so we need to remove the duplicates
    unique_rows = np.unique(rows)
    # get the costs of the non symmetric rows
    unpair_costs = costs.iloc[unique_rows, unique_rows]
    # fill diagonal with np.nan
    filled_costs = unpair_costs.values
    np.fill_diagonal(filled_costs, np.nan)
    unpair_costs = pd.DataFrame(filled_costs, index=unpair_costs.index, columns=unpair_costs.columns)
    # create a matrix to store the symmetric results
    symetric_outs = pd.DataFrame(np.zeros((size, size)), index=costs.index, columns=costs.columns)

    # assign the symmetric ones (coming from assigment problem)
    for row, col in zip(rows_one, cols_one):
        # get row and col pandas indexes
        row_index = costs.index[row]
        col_index = costs.index[col]
        symetric_outs.loc[row_index, col_index] = 1

    # greed algorithm to assign the perfect matches and force outs be symmetric (AxB => BxA)
    for i in range(int(len(unique_rows) / 2)):
        row, col = np.unravel_index(np.nanargmin(unpair_costs), unpair_costs.shape)
        # get row and col pandas indexes
        row_index = unpair_costs.index[row]
        col_index = unpair_costs.index[col]
        # assigned to symetric_outs
        symetric_outs.loc[row_index, col_index] = 1
        symetric_outs.loc[col_index, row_index] = 1
        # deleted row and col
        unpair_costs = unpair_costs.drop([row_index, col_index], axis=0)
        unpair_costs = unpair_costs.drop([row_index, col_index], axis=1)

    return symetric_outs.values
