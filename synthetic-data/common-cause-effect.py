#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Parameters
num_variables = 20
num_time_points = 1000
time_window = (1, 3)
impact_value = 5

# Initialize the dataset with noise variables and binary variables without causes
np.random.seed(0)  # For reproducibility
data = np.random.choice([0, 1], size=(num_variables, num_time_points))

# Impact function, which is a constant in this case
def I(c, e):
    return impact_value if c else -impact_value

# Common cause structure: variable 1 causing variables 2 to 4
for t in range(time_window[1], num_time_points):
    for e in range(2, 5):  # Variables 2 to 4
        # Sum the impact of variable 1 within the time window [t-3, t-1]
        data[e, t] = sum(I(data[1, t - lag], data[e, t]) for lag in range(*time_window))

# Common effect structure: variables 1 and 2 causing variables 3 and 4
for t in range(time_window[1], num_time_points):
    for e in range(3, 5):  # Variables 3 and 4
        # Sum the impact of variables 1 and 2 within the time window [t-3, t-1]
        data[e, t] = sum(I(data[c, t - lag], data[e, t]) for c in range(1, 3) for lag in range(*time_window))

# Discretizing the data for evaluation with Banjo and MSBVAR
# We create bins for positive, negative, and zero values
binned_data = np.digitize(data, bins=[-np.inf, -0.01, 0.01, np.inf]) - 2  # This will create bins -1, 0, 1

# Save the dataset to a CSV file
data_path = "./data/common_cause_effect_dataset.csv"
np.savetxt(data_path, binned_data.transpose(), delimiter=",", fmt='%d')

# Assuming the dataset has been generated and saved as 'common_cause_effect_dataset.csv'
# Load the dataset
data = pd.read_csv(data_path, header=None)

# Transpose the data for plotting (variables on columns, timepoints on rows)
data = data.transpose()

# Set the style of the visualization
sns.set(style="whitegrid")

# Plotting a subset of the variables over time
plt.figure(figsize=(15, 8))

# Select a subset of variables to plot for clarity, here variables 1 to 4
for i in range(1, 5):
    plt.plot(data.index, data[i], label=f'Variable {i}')

plt.title('Subset of Simulated Common Cause and Effect Dataset Over Time')
plt.xlabel('Timepoints')
plt.ylabel('Value')
plt.legend()
plt.show()

# %%
