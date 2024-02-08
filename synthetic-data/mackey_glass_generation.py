import numpy as np
import matplotlib.pyplot as plt
sample_length = 512
delta = 20
C = 0.2
K = 5
sequences = []
for i in range(K):
    sequences.append(1.2 * np.ones(delta) + 0.2 * \
                                    (np.random.rand(delta) - 0.5))
"""y = sequences                                    
x = [i for i in range(delta)]

plt.subplot(5,1,1)
plt.plot(x,y[0])
plt.subplot(5,1,2)
plt.plot(x,y[1])
plt.subplot(5,1,3)
plt.plot(x,y[2])
plt.subplot(5,1,4)
plt.plot(x,y[3])
plt.subplot(5,1,5)
plt.plot(x,y[4])
plt.show()"""

"""C = [np.zeros(K)for _ in range(K)]
for i in range(K):
    C[i][i]=0.2"""
ci = 0.2
cj = 0.4
C = [[ci,cj,0,0,0],
    [0,ci,cj,0,0],
    [0,cj,ci,cj,0],
    [0,0,cj,ci,0],
    [0,0,0,cj,ci]]

for timestep in range(sample_length):
    for i in range(K):
        newi = 0.9*sequences[i][-1]
        coupled_sum = 0
        for j in range(K):
            jtau = sequences[j][-delta]
            coupled_sum+= C[i][j]*jtau/(1+jtau**10)
        newi+=coupled_sum
        sequences[i] = np.append(sequences[i],newi)

x = [i for i in range(sample_length)]
y = sequences
for k in range(K):
    plt.subplot(5,1,k+1)
    plt.plot(x,sequences[k][delta:])
plt.show()
