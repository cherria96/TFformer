# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:10:53 2024

@author: gina1
"""

import numpy as np
import matplotlib.pyplot as plt


K = 5           # number of features
C = 0.2         # coupling strength
T = 512         # time (length of time series data)

# Don't know what the start value should be (when t = 0, 1 should be)

X = np.zeros((K, T))

# first set values for i = 1, K (0, K-1)
for tdx in range(2, T):
    X[0, tdx] = 1.4 - np.power(X[0, tdx-1],2) + 0.3*X[0, tdx-2]
    X[-1, tdx] = 1.4 - np.power(X[-1, tdx-1],2) + 0.3*X[-1, tdx-2]
    
for tdx in range(2, T):
    for idx in range(1, K-1):
        X[idx, tdx] = 1.4 - np.power(0.5*C*(X[idx-1,tdx-1]+X[idx+1,tdx-1]) + (1-C)*X[idx,tdx-1], 2) + 0.3*X[idx, tdx-2]

for idx in range(K):
    plt.figure(figsize=(16,8))
    plt.plot(X[idx].T)
    plt.show()

