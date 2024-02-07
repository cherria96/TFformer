#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Lorenz system parameters
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

# Coupling strength (c = 0 for uncoupled, set to other values for coupled systems)
c = 0.1  # Example value, can be changed as needed

# Define the Lorenz system with coupling
def lorenz_system(t, state, sigma, beta, rho, c):
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = state
    dx1dt = sigma * (y1 - x1)
    dy1dt = x1 * (rho - z1) - y1
    dz1dt = x1 * y1 - beta * z1

    dx2dt = sigma * (y2 - x2) + c * (x1 - x2)
    dy2dt = x2 * (rho - z2) - y2
    dz2dt = x2 * y2 - beta * z2

    dx3dt = sigma * (y3 - x3) + c * (x2 - x3)
    dy3dt = x3 * (rho - z3) - y3
    dz3dt = x3 * y3 - beta * z3

    return [dx1dt, dy1dt, dz1dt, dx2dt, dy2dt, dz2dt, dx3dt, dy3dt, dz3dt]

# Initial conditions for each of the three systems
initial_state = np.random.random(9)

# Time points to solve the system on
t_span = (0, 25)  # From 0 to 25 seconds
t_eval = np.arange(t_span[0], t_span[1], 0.01)  # Evaluate every 0.01 seconds

# Solve the Lorenz system
sol = solve_ivp(lorenz_system, t_span, initial_state, args=(sigma, beta, rho, c), t_eval=t_eval, rtol=1e-10, atol=1e-10)
np.save("./data/Lorenz_time.npy", sol.t)
np.save("./data/Lorenz_solution.npy", sol.y)
# Plotting the results for X1, X2, X3
plt.figure(figsize=(14, 5))
plt.plot(sol.t, sol.y[0], label='X1')
plt.plot(sol.t, sol.y[3], label='X2')
plt.plot(sol.t, sol.y[6], label='X3')
plt.title('System 6: Three Coupled Lorenz Systems')
plt.xlabel('Time')
plt.ylabel('X values')
plt.legend()
plt.show()

# %%
