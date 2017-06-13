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
    frame_dict = {}
    for channel in range(len(channels)):
        data_array = []
        for num in range(event_count):
            row = []
            for num2 in range(channel_count):
                if num2 == channel:
                    row.append(np.arcsinh(np.random.lognormal(mean=3)))
                else:
                    row.append(np.arcsinh(np.random.lognormal(mean=0)))
            data_array.append(row)
        data_array = np.array(data_array)
        data_frame = pd.DataFrame(data_array)
        data_frame.columns = channels
        frame_dict[channels[channel]] = DataSet(data_frame=data_frame)
        #print(data_frame)
    return frame_dict

if __name__ == '__main__':
    spillover = [[0.75,0],
                 [0.25,1]]
    data_dict = generate_samples(2,1000)
    '''
    data_dict = {}
    for channel, path in file_ref.items():
        print(channel)
        data_dict[channel] = DataSet(path='controls/'+path)
        data_dict[channel].data_frame = data_dict[channel].data_frame.iloc[:1000,:]
        print(data_dict[channel].data_frame)
        data_dict[channel].columns = list(data_dict.keys())
    #create_grid(data_dict)
    #print(data_dict['ch1'].find_mutual_info('ch1','ch2'))
    '''
    for data_set in data_dict.values():
        data_set.apply(spillover)
    create_grid(data_dict)

    #print(minimize_mutual_info(data_dict['ch2'], 'ch2', 'ch1'))
    spillover = construct_ideal_matrix(data_dict, resolution=0.01)
    for data_set in data_dict.values():
        data_set.apply(alg.inv(spillover))
    print(spillover)

    create_grid(data_dict)
    #print(data_dict['ch1'].data_frame)
    #print(data_dict['ch1'].find_mutual_info('ch2', 'ch2'))