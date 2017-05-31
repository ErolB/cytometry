from gate import PolygonGate, DataSet
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import scipy.linalg as alg
from compensation import *
import pandas as pd
import copy

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
            #plt.xlim((-10,10))
            #plt.ylim((-10,10))
            # labeling axes
            if ((channel_count-counter) < counter):
                sub.set_xlabel(field2)
            if (counter%channel_count == 1):
                sub.set_ylabel(field1)
            x = data_set.data_frame[field2].values[:1000]
            y = data_set.data_frame[field1].values[:1000]
            # scaling
            x = [np.arcsinh(item / c) for item in x]
            y = [np.arcsinh(item/c) for item in y]
            # generate plot
            xy = np.vstack((x, y))
            #print(xy)
            print(counter)
            try:
                z = gaussian_kde(xy)(xy)
                sub.scatter(x, y, c=np.log10(z))
                figure.add_subplot(sub)
            except:
                pass  # diagonals are left blank
            counter += 1
    plt.show()

def generate_samples(channel_count, event_count):
    channels = ['ch'+str(num) for num in range(1, channel_count+1)]
    frame_list = []
    for channel in range(len(channels)):
        data_array = []
        for num in range(event_count):
            row = []
            for num2 in range(channel_count):
                if num2 == channel:
                    row.append(np.random.lognormal(mean=np.log(1000)))
                else:
                    row.append(np.random.lognormal(mean=np.log(50)))
            data_array.append(row)
        data_array = np.array(data_array)
        data_frame = pd.DataFrame(data_array)
        data_frame.columns = channels
        frame_list.append(data_frame)
        #print(data_frame)
    return frame_list

if __name__ == '__main__':
    spillover = [[0.73, 0.10, 0.05],
                 [0.22, 0.85, 0.02],
                 [0.05, 0.05, 0.93]]
    print(alg.inv(spillover))
    comp = alg.inv(spillover)
    frames = generate_samples(3,1000)
    #data_dict = {'ch1': DataSet(data_frame=frames[0]), 'ch2': DataSet(data_frame=frames[1]), 'ch3': DataSet(data_frame=frames[2])}
    data_dict = {}
    for index, channel in enumerate(frames[0].columns):
        data_dict[channel] = DataSet(data_frame=frames[index])
        data_dict[channel].columns = frames[0].columns
    #create_grid(data_dict)
    #print(data_dict['ch1'].find_mutual_info('ch1','ch2'))
    for data_set in data_dict.values():
        data_set.apply(spillover)
    #print(minimize_mutual_info(data_dict['ch2'], 'ch2', 'ch1'))
    print(construct_ideal_matrix(data_dict))
    create_grid(data_dict)
    #print(data_dict['ch1'].data_frame)
    #print(data_dict['ch1'].find_mutual_info('ch2', 'ch2'))