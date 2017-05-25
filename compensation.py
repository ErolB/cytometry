import numpy as np
import fcswrite
from scipy.integrate import quad
from scipy.stats import gaussian_kde
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

def save_as_fcs(path, data_frame):
    fcswrite.write_fcs(path, data_frame.columns.values, data_frame.values)

def apply_matrix(data_matrix, comp_matrix):
    return np.dot(data_matrix, comp_matrix)

def mutual_info(x_data, y_data):
    # define limits for integration
    x_min = min(x_data)
    x_max = max(x_data)
    y_min = min(y_data)
    y_max = max(y_data)
    # define kernel density functions
    p_x = gaussian_kde(x_data)
    p_y = gaussian_kde(y_data)
    p_xy = gaussian_kde(np.vstack((x_data, y_data)))
    # perform integration
    def f(x,y):
        return p_xy((x,y)) * np.log(p_xy((x,y)) / p_x(x)*p_y(y))

    def f2(y):
        def f3(x):
            return f(x,y)
        return quad(f3, x_min, x_max)[0]

    return quad(f2, y_min, y_max)[0]


if __name__ == '__main__':
    x=[1,2,3]
    y=[2,3,-99]
    print(mutual_info(x,y))