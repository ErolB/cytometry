from gate import PolygonGate, DataSet
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import scipy.linalg as alg
from compensation import load_matrix, apply_matrix
import pandas as pd

file_ref = {'BV421-A': 'Single_001_Ly6C_BV421_002.fcs', 'BV510-A': 'Single_001_MHCII_BV480_003.fcs', 'PE-Cy7-A': 'Single_001_CD11b_PECy7_007.fcs',
            'APC-A': 'Single_001_EdU_APC_008.fcs', 'PE ( 561 )-A': 'Single_001_Ly6G_PE_006.fcs',
            'BUV396-A': 'Single_001_CXCR4_BUV496_001.fcs', 'BV711-A': 'Single_001_CD115_BV711_004.fcs', 'FITC-A': 'Single_001_CD117_BB515_005.fcs'}

def create_grid(data_dict, c=10):
    figure = plt.figure()
    channel_count = len(file_ref)
    counter = 1
    for field1 in file_ref.keys():
        data_set = data_dict[field1]
        for field2 in file_ref.keys():
            sub = plt.subplot(channel_count, channel_count, counter)
            if ((channel_count-counter) < counter):
                sub.set_xlabel(field2)
            if (counter%channel_count == 1):
                sub.set_ylabel(field1)
            x = data_set.data_frame[field1].values[:1000]
            x = [np.arcsinh(item/c) for item in x]
            y = data_set.data_frame[field2].values[:1000]
            y = [np.arcsinh(item/c) for item in y]
            xy = np.vstack((x, y))
            print(counter)
            try:
                z = gaussian_kde(xy)(xy)
                sub.scatter(x, y, c=np.log10(z))
                figure.add_subplot(sub)
            except:
                pass  # diagonals are left blank
            counter += 1
    plt.show()

if __name__ == '__main__':
    name, fields, spill = load_matrix('CompManual2')
    comp = alg.inv(spill)
    data_dict = {name: DataSet('controls/'+file_ref[name]) for name in file_ref.keys()}
    create_grid(data_dict)
    for item in data_dict:
        new_frame = pd.DataFrame(apply_matrix(data_dict[item].data_frame[fields], comp))
        new_frame.columns = fields
        data_dict[item].data_frame = new_frame
    create_grid(data_dict)