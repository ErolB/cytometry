import numpy as np
import fcswrite
from scipy import linalg as alg
from scipy.integrate import quad
from scipy.stats import gaussian_kde
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

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

def minimize_mutual_info(data_set, field1, field2, resolution=0.01):
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
        #new_row *= (1-num)/sum(new_row)
        new_row[column] = num
        new_spill[row] = new_row
        current_set = copy.deepcopy(data_set)
        current_set.apply(alg.inv(new_spill))
        results.append(current_set.find_mutual_info(field1,field2))
        #print(new_spill)
    #d_results = [results[i + 1] - results[i] for i in range(len(results) - 1)]
    ideal = results.index(max(results)) * resolution
    plt.plot(results)
    plt.show()
    plt.plot(results)
    plt.show()
    return(ideal)

def construct_ideal_matrix(data_dict):
    matrix = np.diag(np.ones(len(data_dict)))
    channels = list(data_dict.keys())
    for row in range(len(channels)):
        for column in range(len(channels)):
            if row != column:
                entry = minimize_mutual_info(data_dict[channels[row]], channels[row], channels[column])
                matrix[row][column] = entry
    # normalize diagonal
    for i in range(len(channels)):
        matrix[i,i] = 2 - sum(matrix[:,i])
    return matrix

if __name__ == '__main__':
    x=[1,2,3]
    y=[2,3,-99]