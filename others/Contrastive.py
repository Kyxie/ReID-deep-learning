# This code is for drawing contrastive loss
# Engineer: Kunyang Xie
# Last Update: 8/3/2021

import matplotlib.pyplot as plt
import numpy as np

def contrastive(d, y, margin):
    Max = np.array([margin-d, 0])
    L = y * pow(d, 2) + (1 - y) * pow(np.max(Max), 2)
    return L

if __name__ == '__main__':
    y = np.array([1, 0])
    margin = 1.25
    d = np.linspace(0, 2, 100)
    L_same = list(range(100))
    L_diff = list(range(100))
    for i in range(len(d)):
        L_same[i] = contrastive(d[i], y[0], margin)
        L_diff[i] = contrastive(d[i], y[1], margin)
    
    plt.figure('Contrastive Loss')
    plt.plot(d, L_same)
    plt.plot(d, L_diff, linestyle='dashed')
    plt.legend(['Same', 'Different'])
    plt.xlabel('d (distance)')
    plt.ylabel('L (Loss)')
    plt.title('Contrastive Loss')
    plt.annotate('margin = 1.25', xy=(1.25, 0), xytext=(1.5,0.5), arrowprops=dict(facecolor='black', shrink=0.01))
    plt.show()