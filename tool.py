import utils
import visualize
import compensation
import json
import os
import numpy as np

data_file = 'data.json'

if __name__ == '__main__':
    # load channel-to-file mapping
    json_file = open(data_file, 'r')
    file_ref = json.loads(json_file.read())
    # load data
    data = {}
    for channel, file_name in file_ref.items():
        data[channel] = utils.DataSet(path=file_name)
    # apply gating
    if (input('apply gating? (y/n)').lower() == 'y'):
        # search for existing gates
        gates = os.listdir('./gates')
        # create new gates if not already in place
        for channel, file_name in file_ref.items():
            file_name = file_name.split('/')[-1]  # extract file name
            file_name = file_name.split('.')[0]  # remove ".fcs"
            print(file_name)
            if file_name in gates:
                if(input('Use existing gate?').lower() == 'y'):
                    gate = utils.PolygonGate(path='./gates/'+file_name)
                    data[channel] = gate.apply(data[channel])
                    continue
            # create new gate
            gate = utils.PolygonGate(channel_list=['FSC-A','SSC-A'])
            gate.define_vertices(data[channel].data_frame)
            gate.save_vertices('./gates/'+file_name)
            data[channel] = gate.apply(data[channel])
    # generate compensation matrix
    channels = list(file_ref.keys())
    for data_set in data.values():
        data_set.select_channels(channels)
    visualize.create_grid(data)
    #spill_matrix = compensation.construct_ideal_matrix(data)

    spill_matrix = [[1, 0.06, 0., 0.01, 0.01, 0.01, 0.05, 0.01],
                 [0.41, 1, 0., 0., 0.01, 0.02, 0.07, 0.08],
                 [0.01, 0.02, 1, 0.01, 0.09, 0.01, 0.04, 0.02],
                 [0.01, 0.02, 0.01, 1, 0.01, 0.01, 0.07, 0.01],
                 [0.02, 0.02, 0., 0.02, 1, 0.02, 0.05, 0.03],
                 [0.43, 0.36, 0.12, 0.93, 0.31, 1, 0.38, 0.23],
                 [0.07, 0.09, 0.01, 0.05, 0.03, 0.04, 1, 0.09],
                 [0.06, 0.07, 0.01, 0.08, 0.05, 0.06, 0.14, 1]]

    print(spill_matrix)
    # compensate and display
    comp_matrix = np.linalg.inv(spill_matrix)
    for data_set in data.values():
        data_set.apply(comp_matrix)
    visualize.create_grid(data)

