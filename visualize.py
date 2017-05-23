from gate import PolygonGate, DataSet
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def create_grid(data_set):
    figure = plt.figure()
    channel_count = len(data_set.channel_list)
    for num in range(1, channel_count**2):
        sub = plt.subplot(channel_count, channel_count, num)
        x = data_set.data_frame[data_set.channel_list[num%channel_count]].values[:500]
        y = data_set.data_frame[data_set.channel_list[int(num/channel_count)]].values[:500]
        xy = np.vstack((x, y))
        print(num)
        try:
            z = gaussian_kde(xy)(xy)
            sub.scatter(x, y, c=np.log10(z))
            figure.add_subplot(sub)
        except:
            pass
    plt.show()

if __name__ == '__main__':
    data = DataSet('test.fcs')
    create_grid(data)