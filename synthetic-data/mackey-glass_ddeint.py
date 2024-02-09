#%%
import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint

# Model parameters
K = 5  # Number of variables
delay = 20
delays = np.array([delay] * K)  # Delays Δ1, Δ2, and Δ3

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
    Ylag = Y(t-delay)
    for j in range(K):
        interaction_sum = 0
        for i in range(K):
            interaction_sum += Cij[i, j] * Ylag[i] / (1 + Ylag[i]**10)
        dxdt[j]= -0.1 * Y(t)[j] + interaction_sum    
    return dxdt

# Define the history function for all variables
def values_before_zero(t):
    # Return the initial condition for all t <= 0
    return np.array([0.5 for _ in range(K)])


# Time points for the integration
t_start = 0
t_end = 4096
t_points = np.linspace(t_start, t_end, t_end)

# Run simulations with different initial conditions
sol = ddeint(mackey_glass, values_before_zero, t_points)

output_path = "./data/mackey-glass.npy"  # Update with your desired path
np.save(output_path, sol)

# Using ddeint to solve the system
sol = ddeint(mackey_glass, values_before_zero, t_points)
#%%
plt.figure(figsize=(12, 10))
for i in range(K):
  plt.subplot(K, 1, i+1)
  plt.plot(t_points, sol[:, i], label=f'X{i+1}')
plt.xlabel('Time')
plt.ylabel('State variables')
plt.suptitle('Mackey-Glass System with K={} using ddeint'.format(K))
plt.legend()
plt.show()
#%%


# Plot the solution for each variable
plt.figure(figsize=(12, 6))
for i in range(K):
    plt.plot(t_points, sol[:, i], label=f'X{i+1}')
plt.xlabel('Time')
plt.ylabel('State variables')
plt.suptitle('Mackey-Glass System with K={} using ddeint'.format(K))
plt.legend()
plt.show()

# %%
