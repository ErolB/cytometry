import numpy as np
import scipy.linalg as alg
from gate import DataSet

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

def apply_matrix(data_matrix, comp_matrix):
    return np.dot(comp_matrix, data_matrix.transpose()).transpose()

name, fields, spill = load_matrix('CompManual2')
comp = alg.inv(spill)
data = DataSet('test.fcs').data_frame[fields].values
print(data)
print(data.shape, comp.shape)
print(apply_matrix(data,comp))