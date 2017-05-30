import numpy as np
import fcswrite
from scipy import linalg as alg
from scipy.integrate import quad
from scipy.stats import gaussian_kde
import copy
import matplotlib.pyplot as plt

def load_matrix(path):
    matrix = []
    matrix_file = open(path, 'r')
    lines = matrix_file.readlines()
    matrix_name = lines[0].strip()
    fields = lines[1].strip().split('\t')
    for line in lines[2:]:
        matrix.append(line.split())
    final_matrix = np.array(matrix, dtype=np.float32)
    return matrix_name, fields, final_matrix

def save_as_fcs(path, data_frame):
    fcswrite.write_fcs(path, data_frame.columns.values, data_frame.values)

def apply_matrix(data_matrix, comp_matrix):
    return np.dot(data_matrix, comp_matrix)

def minimize_mutual_info(data_set, field1, field2, resolution=0.02):
    results = []
    # find correct entry in the matrix
    row = list(data_set.data_frame.columns).index(field1)
    column = list(data_set.data_frame.columns).index(field2)
    # calculate mutual information for all values
    spillover_matrix = np.diag(np.ones(len(data_set.data_frame.columns)))
    for num in np.arange(0, 1, resolution):
        new_spill = spillover_matrix.copy()
        new_row = spillover_matrix[row]
        #new_row[column] = 0
        #new_row *= (1-num)
        new_row[column] = num
        new_spill[row] = new_row
        original_frame = copy.copy(data_set.data_frame)
        data_set.apply(new_spill)
        results.append(data_set.find_mutual_info(field1,field2))
        data_set.data_frame = original_frame
    ideal = results.index(min(results)) * resolution
    new_spill = spillover_matrix.copy()
    new_row = spillover_matrix[row]
    new_row[column] = ideal
    #new_row /= float(sum(new_row))
    new_spill[row] = new_row
    plt.plot(results)
    plt.show()
    return(new_spill)

if __name__ == '__main__':
    x=[1,2,3]
    y=[2,3,-99]