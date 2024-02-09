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
S = 1000        # steps for iteration
num_samples = 1

# randomly generate the initial value
X = np.random.rand(K, T, num_samples)

for ndx in range(num_samples):
    for sdx in range(S):
        for tdx in range(2, T):
            for idx in range(K):
                if idx == 0 or idx == K-1:
                    X[idx, tdx, ndx] = 1.4 - np.power(X[idx, tdx-1, ndx],2) + 0.3*X[idx, tdx-2, ndx]
                else:
                    X[idx, tdx, ndx] = 1.4 - np.power(0.5*C*(X[idx-1,tdx-1, ndx]+X[idx+1,tdx-1, ndx]) + (1-C)*X[idx,tdx-1, ndx], 2) + 0.3*X[idx, tdx-2, ndx]



figs, axs= plt.subplots(K, sharex=True, sharey=True, figsize=(20,8))
for idx in range(K):
    axs[idx].plot(X[idx,:,0].T)