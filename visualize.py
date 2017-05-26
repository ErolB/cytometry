from gate import PolygonGate, DataSet
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import scipy.linalg as alg
from compensation import load_matrix, apply_matrix, save_as_fcs, mutual_info
from sklearn.metrics import mutual_info_score
import pandas as pd
import copy
from traceback import print_exc

file_ref = {'BV421-A': 'Single_001_Ly6C_BV421_002.fcs', 'BV510-A': 'Single_001_MHCII_BV480_003.fcs', 'PE-Cy7-A': 'Single_001_CD11b_PECy7_007.fcs',
            'APC-A': 'Single_001_EdU_APC_008.fcs', 'PE ( 561 )-A': 'Single_001_Ly6G_PE_006.fcs',
            'BUV396-A': 'Single_001_CXCR4_BUV496_001.fcs', 'BV711-A': 'Single_001_CD115_BV711_004.fcs', 'FITC-A': 'Single_001_CD117_BB515_005.fcs'}

def create_grid(data_dict, c=10):
    figure = plt.figure(figsize=(10,10))
    channel_count = len(data_dict)
    counter = 1
    for field1 in data_dict.keys():
        data_set = data_dict[field1]
        for field2 in data_dict.keys():
            sub = plt.subplot(channel_count, channel_count, counter)
            plt.xlim((-10,10))
            plt.ylim((-10,10))
            # labeling axes
            if ((channel_count-counter) < counter):
                sub.set_xlabel(field2)
            if (counter%channel_count == 1):
                sub.set_ylabel(field1)
            # take sample
            x = data_set.data_frame[field2].values[:1000]
            y = data_set.data_frame[field1].values[:1000]
            # scaling
            x = [np.arcsinh(item / c) for item in x]
            y = [np.arcsinh(item/c) for item in y]
            # generate plot
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

def generate_samples(channel_count, event_count, spillover):
    channels = ['ch'+str(num) for num in range(1, channel_count+1)]
    frame_list = []
    for channel_num, channel in enumerate(channels):
        data_array = []
        for num in range(event_count):
            row = []
            for num2 in range(channel_count):
                if num2 == channel_num:
                    row.append(np.random.lognormal())
                else:
                    row.append(0)
            data_array.append(row)
        data_array = np.array(data_array)
        data_array = apply_matrix(data_array, alg.inv(spillover))
        print(data_array)
        data_frame = pd.DataFrame(data_array)
        data_frame.columns = channels
        frame_list.append(data_frame)
    return frame_list

if __name__ == '__main__':

    spillover = [[0.95, 0.03, 0.02], [0.1, 0.5, 0.4], [0.01, 0.1, 0.89]]
    frames = generate_samples(3,100,spillover)
    comp = alg.inv(spillover)
    data_dict = {'ch1': DataSet(data_frame=frames[0]), 'ch2': DataSet(data_frame=frames[1]), 'ch3': DataSet(data_frame=frames[2])}
    fields = ['ch1','ch2','ch3']
    #create_grid(data_dict)

    spill = np.array(np.diag([1,1,1]), dtype=np.float32)
    #fields = [item for item in file_ref.keys()]
    #data_dict = {name: DataSet('controls/'+file_ref[name]) for name in file_ref.keys()}
    #create_grid(data_dict)
    #print(fields)

    for index1, field1 in enumerate(data_dict.keys()):
        for index2, field2 in enumerate(data_dict.keys()):
            mutual_info_list = []
            for term in range(0,100):
                term = float(term) / 100
                print(term, index1, index2)
                new_spill = spill.copy()
                new_row = new_spill[index1]
                new_row[index2] = 0
                factor = sum(new_row)
                if factor:
                    new_row = np.multiply(new_row, (1-term)/factor)
                new_row[index2] = term
                new_spill[index1] = new_row
                try:
                    comp = alg.inv(new_spill)
                except:
                    continue
                new_frame = pd.DataFrame(apply_matrix(data_dict[field1].data_frame.loc[:, fields].values, comp))
                new_frame.columns = fields
                x = new_frame[field2].values
                print(x)
                y = new_frame[field1].values
                # generate plot
                xy = np.vstack((x, y))
                try:
                    mutual_info_list.append(mutual_info(x,y))
                except:
                    print_exc()
            print(field1, field2)
            if mutual_info_list:
                print('showing')
                plt.plot(mutual_info_list)
                plt.show()

