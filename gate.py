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

class FCSData():
    data_frame = None
    channel_list = []

    def __init__(self, path, channels=[]):
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

    def apply_manual_gate(self, channel1, channel2, sample_size=1000):
        # calculate kernel density
        x = self.data_frame[channel1][:sample_size]
        y = self.data_frame[channel2][:sample_size]
        xy = np.vstack((x, y))
        z = gaussian_kde(xy)(xy)
        # create scatter plot
        fig1 = plt.figure(num=1, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
        ax = fig1.add_subplot(111)
        ax.scatter(x, y, c=np.log10(z), s=20, edgecolor='')
        # detect clicks
        ax.set_title('click to build Singlet gate')
        line, = ax.plot([3000], [0])  # empty line
        linebuilder = LineBuilder(line)
        plt.show()
        # select data within gate
        verts = np.transpose([linebuilder.xs, linebuilder.ys])
        path1 = Path(verts)
        data = np.transpose([x, y])
        index = path1.contains_points(data)  # Gate in DNA content for singlets
        # return new data set
        trimmed_data = pd.DataFrame(self.data_frame[:].values[index])
        trimmed_data.columns = self.channel_list
        x = trimmed_data[channel1][:sample_size]
        y = trimmed_data[channel2][:sample_size]
        xy = np.vstack((x, y))
        z = gaussian_kde(xy)(xy)
        fig1 = plt.figure(num=1, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
        ax = fig1.add_subplot(111)
        ax.scatter(x, y, c=np.log10(z), s=20, edgecolor='')
        plt.show()
        return trimmed_data


if __name__ == '__main__':
    dataset = FCSData('test.fcs')
    dataset.apply_manual_gate('FSC-A', 'FSC-H')

