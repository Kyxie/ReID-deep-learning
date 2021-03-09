# This code aims to draw 3 activation function
# Engineer: Kunyang Xie
# Last Update: 8/3/2021

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def tanh(x):
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 / s2
    return s

def relu(x):
    s = np.where(x < 0, 0, x)
    return s

if __name__ == '__main__':
    plt.figure('Sigmoid')
    x = np.linspace(-10, 10, 1000)
    s_sig = sigmoid(x)
    plt.plot(x, s_sig)
    plt.xlabel('x')
    plt.ylabel('y = sigmoid(x)')
    plt.title('Sigmoid(x)')
    plt.show()

    plt.figure('Tanh')
    s_tanh = tanh(x)
    plt.plot(x, s_tanh)
    plt.xlabel('x')
    plt.ylabel('y = tanh(x)')
    plt.title('Tanh(x)')
    plt.show()

    plt.figure('ReLU')
    s_relu = relu(x)
    plt.plot(x, s_relu)
    plt.xlabel('x')
    plt.ylabel('y = ReLU(x)')
    plt.title('ReLU(x)')
    plt.show()