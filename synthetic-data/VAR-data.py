#%%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_samples = 100
num_time_points = 1000  # Example length of the time series
std_devs = [1, 0.2, 0.3]  # Standard deviations for theta_t, eta_t, and epsilon_t

def VAR1():
    # Initialize time series for each variable
    num_variables = 3
    X = np.zeros((num_variables, num_time_points, num_samples))

    # Generate noise processes
    theta_t = np.random.normal(0, std_devs[0], (num_time_points, num_samples))
    eta_t = np.random.normal(0, std_devs[1], (num_time_points, num_samples))
    epsilon_t = np.random.normal(0, std_devs[2], (num_time_points, num_samples))

    # Apply VAR(1) model relationships
    for t in range(1, num_time_points):
        X[0, t] = theta_t[t]
        X[1, t] = 0.5 * X[2, t-1] + X[0, t-1] + eta_t[t]
        X[2, t] = X[1, t-1] + epsilon_t[t]

    # Transpose the data to have samples as the first dimension
    X = X.transpose(2, 1, 0) # (num_samples, num_time_points, num_variables)

    # At this point, X is a 3D array where:
    # - The first dimension corresponds to different realizations (samples)
    # - The second dimension corresponds to time points
    # - The third dimension corresponds to variables X1, X2, X3

    # Visualize the time series for the first sample
    sample_index = 0  # First sample for visualization
    time_series_X1 = X[0, :100, sample_index]
    time_series_X2 = X[1, :100, sample_index]
    time_series_X3 = X[2, :100, sample_index]

    # Plotting
    plt.figure(figsize=(20, 6))
    plt.plot(time_series_X1, label='$X_1(t)$')
    plt.plot(time_series_X2, label='$X_2(t)$')
    plt.plot(time_series_X3, label='$X_3(t)$')
    plt.title('System 1: VAR(1) Time Series Visualization')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    np.save("./data/VAR1_dataset.npy", X)

def VAR4():
    # System 3: VAR(4) process in five variables
    num_variables = 5
    num_time_points = 1000  # Adjust as needed
    num_samples = 100  # Number of realizations

    # Initialize the dataset with zeros
    data = np.zeros((num_variables, num_time_points, num_samples))

    # Generate independent Gaussian white noise processes for each variable
    noise_std = 1
    epsilon = np.random.normal(0, noise_std, (num_variables, num_time_points, num_samples))

    # Apply the VAR(4) model relationships
    # Coefficients are as per the system's description
    for t in range(4, num_time_points):
        for sample in range(num_samples):
            # X1, t
            data[0, t, sample] = 0.4 * data[0, t-1, sample] - 0.5 * data[0, t-2, sample] + 0.4 * data[4, t-1, sample] + epsilon[0, t, sample]
            # X2, t
            data[1, t, sample] = 0.4 * data[1, t-1, sample] - 0.3 * data[0, t-4, sample] + 0.4 * data[4, t-2, sample] + epsilon[1, t, sample]
            # X3, t
            data[2, t, sample] = 0.5 * data[2, t-1, sample] - 0.7 * data[2, t-2, sample] - 0.3 * data[4, t-3, sample] + epsilon[2, t, sample]
            # X4, t
            data[3, t, sample] = 0.8 * data[3, t-3, sample] + 0.4 * data[0, t-2, sample] + 0.3 * data[1, t-3, sample] + epsilon[3, t, sample]
            # X5, t
            data[4, t, sample] = 0.7 * data[4, t-1, sample] - 0.5 * data[4, t-2, sample] - 0.4 * data[3, t-1, sample] + epsilon[4, t, sample]

    # The generated data is a 3D array with dimensions corresponding to variables, timepoints, and samples
    # To proceed with analysis, you might want to consider one sample at a time or average across samples
    # Visualize the time series for the first sample
    sample_index = 0  # First sample for visualization
    time_series_data = data[:, :100, sample_index]

    # Plotting
    plt.figure(figsize=(20, 8))
    for i in range(num_variables):
        plt.plot(time_series_data[i], label=f'$X_{i+1}(t)$')

    plt.title('System 3: VAR(4) Time Series Visualization')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    np.save("./data/VAR4_dataset.npy", data.transpose(2,1,0))


# %%
if __name__ == "__main__":
    VAR1()
    VAR4()

# %%
