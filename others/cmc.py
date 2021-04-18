# This code aims to draw cmc curve
# Engineer: Kunyang Xie
# Last Update: 16/4/2021

import matplotlib.pyplot as plt
import numpy as np

rank = [1, 5, 10, 20]
market1501 = [76.3, 91.8, 93.7, 95.9]
UESTC = [80.0, 93.3, 100.0, 100.0]

def cmc(hor, ver, data):
    plt.figure('{}'.format(data))
    plt.plot(hor, ver)
    plt.xlabel('rank@')
    plt.ylabel('%')
    plt.title("CMC Curve of {}".format(data))
    my_x_ticks = rank
    my_y_ticks = np.arange(75, 100, 2)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    for i in range(0, 4):
        plt.annotate("{}%".format(ver[i]), xy=(hor[i], ver[i]), xytext=(hor[i]+1, ver[i]-3), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.grid()
    plt.show()

if __name__ == '__main__':
    cmc(rank, market1501, 'Market1501')
    cmc(rank, UESTC, 'UESTC')