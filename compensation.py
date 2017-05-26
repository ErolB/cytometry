import numpy as np
import fcswrite
from scipy.integrate import dblquad
from scipy.stats import gaussian_kde
from gate import DataSet
from sklearn.metrics import mutual_info_score

file_ref = {'BV421-A': 'Single_001_Ly6C_BV421_002.fcs', 'BV510-A': 'Single_001_MHCII_BV480_003.fcs', 'PE-Cy7-A': 'Single_001_CD11b_PECy7_007.fcs',
            'APC-A': 'Single_001_EdU_APC_008.fcs', 'PE ( 561 )-A': 'Single_001_Ly6G_PE_006.fcs',
            'BUV396-A': 'Single_001_CXCR4_BUV496_001.fcs', 'BV711-A': 'Single_001_CD115_BV711_004.fcs', 'FITC-A': 'Single_001_CD117_BB515_005.fcs'}

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
    x_min = int(min(x_data))
    x_max = int(max(x_data))
    y_min = int(min(y_data))
    y_max = int(max(y_data))
    # define kernel density functions
    p_x = gaussian_kde(x_data)
    p_y = gaussian_kde(y_data)
    p_xy = gaussian_kde(np.vstack((x_data, y_data)))
    # perform integration
    def f(x,y):
        output = p_xy((x,y)) * np.log(p_xy((x,y)) / (p_x(x)*p_y(y)))
        if np.isnan(output):
            output = 0
        return output
    summation = 0
    for x in np.arange(x_min,x_max, float(x_max-x_min)/25):
        for y in np.arange(y_min,y_max, float(y_max-y_min)/25):
            summation += f(x,y)
    return summation


if __name__ == '__main__':
    print(mutual_info([1,2,558],[9,3,-6]))