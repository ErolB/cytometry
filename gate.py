import pandas as pd
import numpy as np
import fcsparser
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pylab import *
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib

class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes != self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

class DataSet():
    data_frame = None
    channel_list = []

    def __init__(self, path=None, data_frame=None, channels=[]):
        if path:
            metadata, data = fcsparser.parse(path, reformat_meta=True)
            if not channels:  # use all channels by default
                channels = list(metadata['_channels_']['$PnN'].values)
            # construct data frame
            self.data_frame = data[channels[0]]  # initialize
            for field in channels[1:]:
                self.data_frame = pd.concat([self.data_frame, data[field]], axis=1)
            self.data_frame.columns = channels
            # unpack metadata
            self.channel_list = list(metadata['_channels_']['$PnN'].values)
        elif data_frame is not None:
            self.data_frame = data_frame
            self.channel_list = data_frame.columns

    # applies composition matrix
    def apply(self, matrix):
        new_data = np.dot(self.data_frame.values, matrix)
        self.data_frame = pd.DataFrame(new_data)
        self.data_frame.columns = self.channel_list

    # calculates the mutual inforamtaion for two columns
    def find_mutual_info(self, field1, field2, resolution=50):
        x_data = self.data_frame[field1].values
        y_data = self.data_frame[field2].values
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
        def f(x, y):
            ans = p_xy((x, y)) * np.log(p_xy((x, y)) / p_x(x) * p_y(y))
            if np.isnan(ans):
                print('warning: invalid')
                return 0
            else:
                #print(p_xy((x, y)) / p_x(x) * p_y(y))
                return ans
        x_interval = float(x_max - x_min) / resolution
        y_interval = float(y_max - y_min) / resolution
        sum = 0
        for x in np.arange(x_min, x_max, x_interval):
            for y in np.arange(y_min, y_max, y_interval):
                sum += f(x, y)
        sum *= (x_interval * y_interval)
        return sum


class Gate():
    channel_list = []

    def __init__(self, channel_list):
        self.channel_list = channel_list


class PolygonGate(Gate):
    vertex_array = None

    def __init__(self, channel_list=[], path=None):
        if channel_list:
            super().__init__(channel_list)
        elif path:
            vertex_file = open(path, 'r')
            vertices = []
            find_channels = True  # specifies that the program is reading the channel list
            for line in vertex_file.readlines():
                line = line.strip()  # remove whitespace and newlines
                if find_channels:
                    self.channel_list = line.split(',')
                    find_channels = False
                    continue
                x, y = line.split(',')
                vertices.append([float(x), float(y)])
            self.vertex_array = np.array(vertices)
            vertex_file.close()

    def apply(self, data_set, sample_size=1000):
        x = data_set.data_frame[self.channel_list[0]][:sample_size]
        y = data_set.data_frame[self.channel_list[1]][:sample_size]
        if self.vertex_array is None:
            self.define_vertices(x,y)
        path1 = Path(self.vertex_array)
        data = np.transpose([x, y])
        index = path1.contains_points(data)  # Gate in DNA content for singlets
        # return new data set
        trimmed_data = pd.DataFrame(data_set.data_frame[:].values[index])
        trimmed_data.columns = data_set.channel_list
        x = trimmed_data[self.channel_list[0]][:sample_size]
        y = trimmed_data[self.channel_list[1]][:sample_size]
        xy = np.vstack((x, y))
        z = gaussian_kde(xy)(xy)
        fig1 = plt.figure(num=1, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
        ax = fig1.add_subplot(111)
        ax.scatter(x, y, c=np.log10(z), s=20, edgecolor='')
        plt.show()
        return DataSet(data_frame=trimmed_data)  # creates new object

    def define_vertices(self, data_frame, sample_size=1000):
        x = data_frame[self.channel_list[0]][:sample_size]
        y = data_frame[self.channel_list[1]][:sample_size]
        xy = np.vstack((x, y))
        z = gaussian_kde(xy)(xy)
        # create scatter plot
        fig1 = plt.figure(num=1, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
        ax = fig1.add_subplot(111)
        ax.scatter(x, y, c=np.log10(z), s=20, edgecolor='')
        # detect clicks
        ax.set_title('click to build Singlet gate')
        line, = ax.plot([0],[0])  # empty line
        linebuilder = LineBuilder(line)
        plt.show()
        # select data within gate
        self.vertex_array = np.transpose([linebuilder.xs, linebuilder.ys])

    def save_vertices(self, path):
        vertex_file = open(path, 'w')
        if self.vertex_array is None:
            print('no vertex set available')
            return
        line = ','.join(self.channel_list) + '\n'
        vertex_file.write(line)  # write list of channels
        for row in self.vertex_array:
            line = str(row[0]) + ',' + str(row[1]) + '\n'
            vertex_file.write(line)
        vertex_file.close()


if __name__ == '__main__':
    dataset = DataSet('test.fcs')
    print(dataset.channel_list)
    gate1 = PolygonGate(channel_list=['FSC-A','SSC-A'])
    gate1.define_vertices(dataset.data_frame)
    gate1.save_vertices('vertex2.txt')
    dataset2 = gate1.apply(dataset)

