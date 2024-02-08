#%%
import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint

# Model parameters
K = 5  # Number of variables
delays = np.array([20] * K)  # Delays Δ1, Δ2, and Δ3

# Connectivity matrix Cij for the specified system
ci = 0.2
cj = 0.4
Cij = np.array([[ci,cj,0,0,0],
    [0,ci,cj,0,0],
    [0,cj,ci,cj,0],
    [0,0,cj,ci,0],
    [0,0,0,cj,ci]])
# Mackey-Glass differential equation
def mackey_glass(Y,t):
    dxdt = np.zeros(K)  
    for j in range(K):
        interaction_sum = 0
        for i in range(K):
            Ylag = Y(t-delays[i])
            interaction_sum += Cij[i, j] * Ylag[i] / (1 + Ylag[i]**10)
        dxdt[j] = -0.1 * Y[j] + interaction_sum    
    return dxdt

# Define the history function for all variables
def values_before_zero(t):
    # Return the initial condition for all t <= 0
    return np.random.rand(K)


# Time points for the integration
t_start = 0
t_end = 4096
t_points = np.linspace(t_start, t_end, 4096)

# Using ddeint to solve the system
sol = ddeint(mackey_glass, values_before_zero, t_points)

# Plot the solution for each variable
plt.figure(figsize=(12, 6))
for i in range(K):
    plt.plot(t_points, sol[:, i], label=f'X{i+1}')
plt.xlabel('Time')
plt.ylabel('State variables')
plt.title('Mackey-Glass System with K=5 using ddeint')
plt.legend()
plt.show()
