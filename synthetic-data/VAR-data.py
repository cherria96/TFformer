#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

# Parameters
num_samples = 1
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
    """
    1. Covariates (C1 ~ C5)
        C1,t depends on C1,t-1 & C1,t-2 & C5,t-1
        C2,t depends on C2,t-1 & C1,t-4 & C5,t-2
        C3,t depends on C3,t-1 & C3,t-2 & C5,t-3
        C4,t depends on C4,t-3 & C1,t-2 & C2,t-3
        C5,t depends on C5,t-1 & C5,t-2 & C3,t-1
    2. Treatments (T1 ~ T2)
        T1,t depends on C3,t-1 & C2,t-2
        T2,t depends on C4,t-1 & C5,t-1 & C2,t-2
    3. Outcomes (O1)
        O1,t depends on T1,t-1 & T2,t-2 & C4,t-1 & S1

    """
    # System 3: VAR(4) process in five variables
    # num_variables = 5
    num_covariates = 5 # C1 ~ C5
    num_static_features = 2 # S1, S2
    num_treatments = 2 # T1 ~ T2
    num_outcomes = 1 # O1

    num_time_points = 1000  # Adjust as needed
    num_samples = 100  # Number of realizations

    # Initialize the dataset with zeros
    # data = np.zeros((num_variables, num_time_points, num_samples))
    covariates = np.zeros((num_covariates, num_time_points, num_samples))
    static_features = np.zeros((num_static_features, num_samples))
    treatments = np.zeros((num_treatments, num_time_points, num_samples))
    outcomes = np.zeros((num_outcomes, num_time_points, num_samples))

    # Generate independent Gaussian white noise processes for each variable
    # epsilon = np.random.normal(0, noise_std, (num_variables, num_time_points, num_samples))
    epsilon_covariates = np.random.normal(0, 1, (num_covariates, num_time_points, num_samples))
    epsilon_treatments = np.random.normal(0, 1, (num_treatments, num_time_points, num_samples))
    epsilon_outcomes = np.random.normal(0, 1, (num_outcomes, num_time_points, num_samples))
    
    # Generate static features
    for sample in range(num_samples):
        static_features[:, sample] = np.random.normal(0, 1, num_static_features)

    # Apply the VAR(4) model relationships
    # Coefficients are as per the system's description
    for t in range(4, num_time_points):
        for sample in range(num_samples):
            # C1, t
            covariates[0, t, sample] = 0.4 * covariates[0, t-1, sample] - 0.5 * covariates[0, t-2, sample] + 0.4 * covariates[4, t-1, sample] + epsilon_covariates[0, t, sample]
            # C2, t
            covariates[1, t, sample] = 0.4 * covariates[1, t-1, sample] - 0.3 * covariates[0, t-4, sample] + 0.4 * covariates[4, t-2, sample] + epsilon_covariates[1, t, sample]
            # C3, t
            covariates[2, t, sample] = 0.5 * covariates[2, t-1, sample] - 0.7 * covariates[2, t-2, sample] - 0.3 * covariates[4, t-3, sample] + epsilon_covariates[2, t, sample]
            # C4, t
            covariates[3, t, sample] = 0.8 * covariates[3, t-3, sample] + 0.4 * covariates[0, t-2, sample] + 0.3 * covariates[1, t-3, sample] + epsilon_covariates[3, t, sample]
            # C5, t
            covariates[4, t, sample] = 0.7 * covariates[4, t-1, sample] - 0.5 * covariates[4, t-2, sample] - 0.4 * covariates[3, t-1, sample] + epsilon_covariates[4, t, sample]
            # T1, t
            treatments[0, t, sample] = 0.6 * covariates[2, t-1, sample] - 0.3 * covariates[1, t-2, sample] + epsilon_treatments[0, t, sample]
            # T2, t
            treatments[1, t, sample] =  0.8 * covariates[3, t-1, sample] + 0.4 * covariates[4, t-1, sample] - 0.6 * covariates[1, t-2, sample] + epsilon_treatments[1, t, sample]
            # O1, t
            outcomes[0, t, sample] = 0.7 * treatments[0, t-1, sample] + 0.4 * treatments[1, t-2, sample]- 0.5 * covariates[3, t-1, sample] + 0.3 * static_features[1, sample] + epsilon_outcomes[0, t, sample]
    # The generated data is a 3D array with dimensions corresponding to variables, timepoints, and samples
    # To proceed with analysis, you might want to consider one sample at a time or average across samples
    # Visualize the time series for the first sample
    data = np.concatenate([treatments, covariates, outcomes], axis =0)
    sample_index = 0  # First sample for visualization
    time_series_data = data[:, :100, sample_index]

    # Plotting
    num_variables = num_covariates + num_outcomes + num_treatments
    plt.figure(figsize=(20, 8))
    for i in range(num_variables):
        plt.plot(time_series_data[i], label=f'$X_{i+1}(t)$')

    plt.title('System 3: VAR(4) Time Series Visualization')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
    data = dict()
    data['treatments'] = treatments
    data['covariates'] = covariates
    data['outcomes'] = outcomes
    data['static_features'] = static_features
    # np.save("./data/VAR4_dataset.npy", data.transpose(2,1,0))
    # Create a new HDF5 file
    with h5py.File('data.h5', 'w') as hdf:
        for key, value in data.items():
            hdf.create_dataset(key, data=value)



# %%
if __name__ == "__main__":
    # VAR1()
    VAR4()

# %%
