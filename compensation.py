import numpy as np
import fcswrite
from scipy import linalg as alg
from scipy.integrate import quad
from scipy.stats import gaussian_kde
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

from analytical import estimate_mutual_info

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

def minimize_mutual_info(data_set, field1, field2, resolution=0.01, upper = 0.9, lower = 0):
    results = []
    # calculate mutual information for all values
    for theta in np.arange(lower, upper, resolution):
        new_spill = [[1-theta, 0],
                     [theta, 1]]
        current_array = [copy.copy(data_set.data_frame[field1]), copy.copy(data_set.data_frame[field2])]
        #current_array = np.log(current_array)
        current_array = np.dot(alg.inv(new_spill), current_array)
        results.append(estimate_mutual_info(current_array))
        #print(new_spill)
    #d_results = [results[i + 1] - results[i] for i in range(len(results) - 1)]
    ideal = results.index(min(results)) * resolution
    plt.plot(results)
    plt.show()
    plt.plot(results)
    plt.show()
    return(ideal)

def construct_ideal_matrix(data_dict, resolution=0.02, upper=0.9, lower=0):
    matrix = np.diag(np.ones(len(data_dict)))
    channels = list(data_dict.keys())
    for row in range(len(channels)):
        for column in range(len(channels)):
            if row != column:
                entry = minimize_mutual_info(data_dict[channels[row]], channels[column], channels[row], resolution=resolution, upper=upper, lower=lower)
                matrix[row][column] = entry
    # normalize diagonal
    for i in range(len(channels)):
        matrix[i,i] = 2 - sum(matrix[:,i])
    return matrix

if __name__ == '__main__':
    x=[1,2,3]
    y=[2,3,-99]