# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Model parameters from the table
A = 3.25  # mV, Average excitatory synaptic gain
B = 22.0  # mV, Average inhibitory synaptic gain
a = 100.0  # s^-1, Membrane average time constant and dendritic tree average time delays
b = 50.0   # s^-1, Membrane average time constant and dendritic tree average time delays
C = 135.0  # Connectivity constant
C1 = C
C2 = 0.8 * C
C3 = 0.25 * C
C4 = 0.25 * C
v0 = 6.0  # mV, Firing threshold
e0 = 2.5  # s^-1, Half-activation rate
r = 0.56  # mV^-1, Steepness of the sigmoid function
ad = 33.0  # s^-1, Average time delay on efferent connection from a population
p0 = 200
# K neural masses (number of populations)
K = 5  # This can be adjusted based on the structure of the network

# Define the simulation parameters
dt = 0.001  # Simulation time step
t_max = 60  # Total time of simulation
time = np.arange(0, t_max, dt)
num_points = len(time)

# Define the white noise input p(t)
mean_rate = (30 + 150) / 2  # Average rate of pulses per second
std_dev = np.sqrt(mean_rate)  # Standard deviation corresponding to pulse rate
p_t = np.random.normal(loc=mean_rate, scale=std_dev, size=num_points)
# std_dev = 0.1*p0
# p_t = p0 + np.random.normal(0, std_dev, K)

# Connectivity matrix Cij for ring structure
Cij = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        if i != j:
            Cij[i, j] = C  # Coupling strength for interconnected neural masses

def sigmoid(v):
    # tanh is a scaled sigmoid function; this should prevent overflow
    return e0 * (1 + np.tanh(r / 2 * v))

def neural_mass_model(t, y, A, B, a, b, C1, C2, C3, C4, Cij, p_t):
    dydt = np.zeros_like(y)  # Ensuring dydt is a flat numpy array with the correct shape
    for j in range(K):
        # Extracting the state variables for the j-th neural mass
        y0, y1, y2, y3, y4, y5, y6, y7 = y[j*8:(j+1)*8]
        
        # Interactions with other neural masses
        interaction_sum = np.clip(sum(Cij[j, i] * y[4 + i*8] for i in range(K) if i != j), -500, 500)
        # p_current = np.interp(t, time, p_t)
        p_current = np.interp(t, np.linspace(0, t_max, num_points), p_t)
        # Differential equations
        dydt[j*8] = y3
        dydt[j*8 + 1] = y4
        dydt[j*8 + 2] = y5
        dydt[j*8 + 3] = A * a * sigmoid(y1-y2) - 2 * a * y3 - a**2 * y0
        # dydt[j*8 + 2] = np.clip(A * a * sigmoid(y1 - y2) - 2 * a * y3 - a**2 * y0, -500, 500)
        dydt[j*8 + 4] = A * a * (p_current + C2 * sigmoid(C1 * y0) + interaction_sum)  - 2 * a * y4 - a**2 * y1
        dydt[j*8 + 5] = B * b * C4 * sigmoid(C3 * y0) - 2 * b * y5 - b**2 * y2
        # dydt[j*8 + 5] = 0  # Assuming y5's derivative is zero as it may represent an external input or other behavior not modeled by a differential equation
        dydt[j*8 + 6] = y7
        dydt[j*8 + 7] = A * ad * sigmoid(y1 - y2) - 2 * ad * y7 - ad**2 * y6

    return dydt


# # Define the parameters as a tuple
# parameters = (A, B, a, b, C1, C2, C3, C4, Cij)

# Wrap the model function to include the noise term
def model_wrapper(t, y):
    return neural_mass_model(t, y, A, B, a, b, C1, C2, C3, C4, Cij, p_t)

# Define the time span and initial conditions as before
t_span = [0, 60]
y0 = np.zeros(8 * K)



# # Call solve_ivp, passing 'parameters' as the 'args' argument
# solution = solve_ivp(
#     fun=lambda t, y: neural_mass_model(t, y, *parameters),
#     t_span=t_span,
#     y0=y0,
#     t_eval=np.linspace(t_span[0], t_span[1], 6000),
#     method='BDF',
#     vectorized=True,
#     rtol=1e-6,
#     atol=1e-9,
#     max_step=1e-3
# )


# Call the ODE solver
solution = solve_ivp(
    fun=model_wrapper,
    t_span=t_span,
    y0=y0,  # Make sure to define your initial conditions vector y0_initial
    t_eval=np.linspace(t_span[0], t_span[1], 6000),
    method='BDF',
    vectorized=True,
    rtol=1e-6,
    atol=1e-9,
    max_step=1e-3
)

# 

# Check if the solver was successful
if not solution.success:
    raise RuntimeError("ODE solver did not converge: " + solution.message)


# After solving, you can plot the results
plt.plot(solution.t, solution.y[0])  # Assuming y[0] is the variable of interest
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Synthetic EEG Data from Neural Mass Model')
plt.show()

# # Extract the results and plot
# synthetic_eeg_data = [solution.y[i] for i in range(0, 8*K, 8)]
# plt.plot(solution.t, synthetic_eeg_data[0])
# plt.title('Synthetic EEG Data from Neural Mass Model')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()
# %%
