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

    def apply_gate(self, channel1, channel2, sample_size=1000):
        dataset = FCSData('test.fcs')
        x = dataset.data_frame[channel1][:sample_size]
        y = dataset.data_frame[channel2][:sample_size]
        xy = np.vstack((x, y))
        z = gaussian_kde(xy)(xy)
        fig1 = plt.figure(num=1, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
        ax = fig1.add_subplot(111)
        ax.scatter(x, y, c=np.log10(z), s=20, edgecolor='')
        plt.show()

if __name__ == '__main__':
    dataset = FCSData('test.fcs')
    dataset.apply_gate('FSC-A', 'FSC-H')

'''
DirectoryContent = os.listdir()
ListOfFiles = []
for ii in range(np.size(DirectoryContent)):
    if DirectoryContent[ii].find('Normalized.FCS') > 0:
        ListOfFiles.append(DirectoryContent[ii])

# for File in ListOfFiles:

File = ListOfFiles[0]
path = DirectoryContent[DirectoryContent.index(File)]
meta, data = fcsparser.parse(path, reformat_meta=True)

IndexOfChannels = np.array([])
CyTOF = pd.DataFrame
CyTOF = data[data.columns[2]]
NumberOfChannels = 0
ListOfChannels = []

for ii in range(0, np.shape(meta['_channels_'])[0]):
    if meta['_channels_']['$PnN'].iloc[ii].find('Pt195Di') >= 0:
        if NumberOfChannels == 1:
            CyTOF = data[data.columns[ii]]
        else:
            CyTOF = pd.concat([CyTOF, data[data.columns[ii]]], axis=1)
        IndexOfChannels = np.append(IndexOfChannels, [ii], axis=0)
        ListOfChannels = np.append(ListOfChannels, 'Live/Dead (Pt195Di)')

    if meta['_channels_']['$PnN'].iloc[ii].find('Di') > 0:
        if (meta['$P' + str(ii + 1) + 'S'].find('_') > 0) \
                & ~(meta['$P' + str(ii + 1) + 'S'].find('Environ') > 0) \
                & ~(meta['$P' + str(ii + 1) + 'S'].find('beads') > 0):
            NumberOfChannels = NumberOfChannels + 1
            if NumberOfChannels == 1:
                CyTOF = data[data.columns[ii]]
            else:
                CyTOF = pd.concat([CyTOF, data[data.columns[ii]]], axis=1)
            IndexOfChannels = np.append(IndexOfChannels, [ii], axis=0)
            jj = meta['$P' + str(ii + 1) + 'S'].find('_')
            ListOfChannels = np.append(ListOfChannels,
                                       meta['$P' + str(ii + 1) + 'S'][jj + 1:] + ' [' + meta['$P' + str(ii + 1) + 'S'][
                                                                                        :jj] + ']')

CyTOF.columns = ListOfChannels


# %%

def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))
    return [event.xdata, event.ydata]


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


# %%

x = CyTOF['DNA1 [191Ir]'].values
y = CyTOF['DNA2 [193Ir]'].values
xy = np.vstack([x[:1000], y[:1000]])
z = gaussian_kde(xy)(xy)

idx = z.argsort()
xplot, yplot, zplot = x[idx], y[idx], z[idx]

fig_numbers = [x.num for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
for index, item in enumerate(fig_numbers):
    if item == 1:
        plt.close(fig1)

fig1 = plt.figure(num=1, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
ax = fig1.add_subplot(111)
ax.scatter(xplot, yplot, c=np.log10(zplot), s=20, edgecolor='')
plt.xlabel('DNA1 [191Ir]')
plt.ylabel('DNA2 [193Ir]')
plt.show()

ax = fig1.add_subplot(111)
ax.set_title('click to build Singlet gate')
line, = ax.plot([3000], [0])  # empty line
linebuilder = LineBuilder(line)
pause(10)
plt.show()

verts = np.transpose([linebuilder.xs, linebuilder.ys])

path1 = Path(verts)
data = np.transpose([x, y])
index = path1.contains_points(data)  # Gate in DNA content for singlets

# %%
Singlet = pd.DataFrame(CyTOF[:].values[index])
Singlet.columns = CyTOF.columns
x = np.log10(Singlet['Live/Dead (Pt195Di)'].values)
y = Singlet['DNA1 [191Ir]'].values
xy = np.vstack([x[:1000], y[:1000]])
z = gaussian_kde(xy)(xy)

idx = z.argsort()
xplot, yplot, zplot = x[idx], y[idx], z[idx]

fig_numbers = [x.num for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
for index, item in enumerate(fig_numbers):
    if item == 2:
        plt.close(fig2)

fig2 = plt.figure(num=2, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
ax = fig2.add_subplot(111)
ax.scatter(xplot, yplot, c=np.log10(zplot), s=20, edgecolor='')
plt.xlabel('Live/Dead (Pt195Di)')
plt.ylabel('DNA1 [191Ir]')
plt.show()

ax2 = fig2.add_subplot(111)
ax2.set_title('click to select live cells')
line, = ax2.plot([0], [0])  # empty line
linebuilder = LineBuilder(line)
pause(10)
plt.show()

verts = np.transpose([linebuilder.xs, linebuilder.ys])

path2 = Path(verts)
data = np.transpose([x, y])
index2 = path2.contains_points(data)  # Gate in live cells

ListOfChannels_noDNA_noLiveDEAD = []
for ii in range(np.shape(ListOfChannels)[0]):
    if (ListOfChannels[ii].find('Live') < 0) & (ListOfChannels[ii].find('DNA') < 0):
        ListOfChannels_noDNA_noLiveDEAD = np.append(ListOfChannels_noDNA_noLiveDEAD, ListOfChannels[ii])

LiveSinglet = pd.DataFrame(Singlet[ListOfChannels_noDNA_noLiveDEAD].values[index2])
LiveSinglet.columns = Singlet[ListOfChannels_noDNA_noLiveDEAD].columns

# %%
fig_numbers = [x.num for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
for index, item in enumerate(fig_numbers):
    if item == 3:
        plt.close(fig3)
fig3 = plt.figure(num=3, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')

NumberOfChannels = np.shape(ListOfChannels_noDNA_noLiveDEAD)[0]

for jj in range(NumberOfChannels):  # range(5): #
    for ii in range(jj):
        ax = fig3.add_subplot(NumberOfChannels, NumberOfChannels, 1 + jj + ii * NumberOfChannels)
        plt.loglog(LiveSinglet[ListOfChannels_noDNA_noLiveDEAD[ii]][0:100],
                   LiveSinglet[ListOfChannels_noDNA_noLiveDEAD[jj]][0:100], '.',
                   markersize=3)

        if ii == 0:
            plt.title(ListOfChannels_noDNA_noLiveDEAD[jj], fontsize=6)
        if jj == ii - 1:
            plt.ylabel(ListOfChannels_noDNA_noLiveDEAD[ii], fontsize=6)
        plt.xlim((1, 1e5))
        plt.ylim((1, 1e5))
        ax.set_xticklabels('')
        ax.set_yticklabels('')
plt.savefig('Two_by_Two.pdf')
'''