"""
This contains functions for estimating the mutual information of a data set and constructing a compensation matrix
"""

import numpy as np
import fcswrite
from scipy import linalg as alg
import copy
import matplotlib.pyplot as plt

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
    # wide search, low resolution
    for theta in np.arange(lower, upper, 0.05):
        new_spill = [[1-theta, 0],
                     [theta, 1]]
        current_array = [copy.copy(data_set.data_frame[field1]), copy.copy(data_set.data_frame[field2])]
        current_array = np.dot(alg.inv(new_spill), current_array)
        current_array = np.arcsinh(current_array)
        results.append(estimate_mutual_info(current_array, resolution=25))
    ideal = results.index(min(results)) * 0.05
    first_pass = results
    #plt.plot(results)
    #plt.show()
    # increased resolution
    results = []
    upper = ideal + 0.1
    lower = ideal - 0.1
    if upper > 1:
        upper = 1
    if lower < 0:
        lower = 0
    for theta in np.arange(lower, upper, 0.01):
        new_spill = [[1 - theta, 0],
                     [theta, 1]]
        current_array = [copy.copy(data_set.data_frame[field1]), copy.copy(data_set.data_frame[field2])]
        current_array = np.dot(alg.inv(new_spill), current_array)
        current_array = np.arcsinh(current_array)
        results.append(estimate_mutual_info(current_array, resolution=50))
    ideal = (results.index(min(results)) * 0.01) + lower
    #plt.plot(results)
    #plt.show()
    '''
    # highest resolution
    results = []
    upper = ideal + 0.02
    lower = ideal - 0.02
    if upper > 1:
        upper = 1
    if lower < 0:
        lower = 0
    for theta in np.arange(lower, upper, 0.002):
        new_spill = [[1 - theta, 0],
                     [theta, 1]]
        current_array = [copy.copy(data_set.data_frame[field1]), copy.copy(data_set.data_frame[field2])]
        current_array = np.dot(alg.inv(new_spill), current_array)
        current_array = np.arcsinh(current_array)
        results.append(estimate_mutual_info(current_array, resolution=100))
    ideal = (results.index(min(results)) * 0.002) + lower
    print(ideal)
    plt.plot(results)
    plt.show()
    '''
    print(ideal)
    return(ideal, first_pass)

def construct_ideal_matrix(data_dict, resolution=0.02, upper=0.9, lower=0, visualize=False):
    info_sets = []
    matrix = np.diag(np.ones(len(data_dict)))
    channels = list(data_dict.keys())
    for row in range(len(channels)):
        for column in range(len(channels)):
            if row != column:
                entry, data = minimize_mutual_info(data_dict[channels[row]], channels[row], channels[column], resolution=resolution, upper=upper, lower=lower)
                matrix[row][column] = entry
                info_sets.append(data)
            else:
                info_sets.append([0])
    # normalize diagonal
    for i in range(len(channels)):
        matrix[i,i] = 2 - sum(matrix[:,i])
    # create plots and heatmap
    if visualize:
        heatmap = np.zeros((len(channels),len(channels)))
        figure = plt.figure(figsize=(10,10))
        counter = 0
        for row in range(len(channels)):
            for column in range(len(channels)):
                sub = plt.subplot(len(channels), len(channels), counter+1)
                if row != column:
                    sub.plot(info_sets[counter])
                figure.add_subplot(sub)
                heatmap[row][column] = min(info_sets[counter])
                counter += 1
        plt.show()
        print(heatmap)
        plt.imshow(heatmap)
    return matrix
