import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Model parameters
A = 3.25
B = 22.0
a = 100.0
b = 50.0
C = 135.0
C1 = C
C2 = 0.8 * C
C3 = 0.25 * C
C4 = 0.25 * C
v0 = 6.0
e0 = 2.5
r = 0.56
ad = 33.0
K = 5  # Number of neural masses
num_variables = 8  # Number of state variables per neural mass
total_variables = K * num_variables

# Connectivity matrix Cij for the ring structure
Cij = np.zeros((K, K)) + C
np.fill_diagonal(Cij, 0)  # No self-interaction

# Sigmoid function
def sigmoid(v):
    return e0 / (1 + np.exp(r * (v0 - v)))

# Neural mass model function
def neural_mass_model(t, y, A, B, a, b, C1, C2, C3, C4, Cij, num_variables):
    yj = y.reshape((K, num_variables))
    dydt = np.zeros_like(yj)
    p_current = p_t_interp(t)  # Interpolate p_t for the current time t

    for j in range(K):
        interaction_sum = sum(Cij[j, i] * yj[i, num_variables-1] for i in range(K) if i != j)

        # Compute derivatives based on the provided equations
        dydt[j, 0] = yj[j, 3]
        dydt[j, 1] = yj[j, 4]
        dydt[j, 2] = yj[j, 5]
        dydt[j, 3] = A * a * sigmoid(yj[j, 1] - yj[j, 2]) - 2 * a * yj[j, 3] - a**2 * yj[j, 0]
        dydt[j, 4] = A * a * (p_current[j] + C2 * sigmoid(C1 * yj[j, 0]) + interaction_sum) - 2 * a * yj[j, 4] - a**2 * yj[j, 1]
        dydt[j, 5] = B * b * C4 * sigmoid(C3 * yj[j, 0]) - 2 * b * yj[j, 5] - b**2 * yj[j, 2]
        dydt[j, 6] = yj[j, 7]
        dydt[j, 7] = A * ad * sigmoid(yj[j, 1] - yj[j, 2]) - 2 * ad * yj[j, 7] - ad**2 * yj[j, 6]

    return dydt.flatten()

# Time span for the simulation
t_span = [0, 25]
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Define the white noise input p(t) for each neural mass
p_t = np.random.normal(loc=(30 + 150) / 2, scale=np.sqrt((30 + 150) / 2), size=(K, len(t_eval)))
p_t_interp = lambda t: np.array([np.interp(t, t_eval, p_t[j]) for j in range(K)])


# Simulation loop
num_samples = 100  # Number of different simulations (samples)
all_simulations = np.zeros((num_samples, len(t_eval), total_variables))  # Array to store all simulations

for i in range(num_samples):
    initial_conditions = np.random.rand(total_variables)
    solution = solve_ivp(lambda t, y: neural_mass_model(t, y, A, B, a, b, C1, C2, C3, C4, Cij, num_variables),
                    t_span, initial_conditions, t_eval=t_eval, method='BDF')
    if solution.success:
        all_simulations[i] = solution.y.T
    else:
        print(f"Integration failed for sample {i}")

# Save the simulation data
np.save('neural_mass_simulations.npy', all_simulations)

#%%
# After solving, you can plot the results
plt.plot(solution.t, solution.y[0], label = 'X1')  # Assuming y[0] is the variable of interest
plt.plot(solution.t, solution.y[1], label = 'X2')  # Assuming y[0] is the variable of interest
plt.plot(solution.t, solution.y[2], label = 'X3')  # Assuming y[0] is the variable of interest

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Synthetic EEG Data from Neural Mass Model')
plt.show()


# %%
