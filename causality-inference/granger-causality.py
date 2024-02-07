#%%
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
class granger_causality:

    def __init__(self, data_path:str, maxlag:int = 4) -> None:
        self.data_path:str = data_path # Path to npy file (num_samples, num_time_points, num_variables)
        self.maxlag: int = maxlag
        self.data, self.data_shape  = self._data_preprocessing()
        self.maxlag = maxlag

    def _data_preprocessing(self):
        data = np.load(self.data_path)
        data_shape = data.shape
        data_mean: np.ndarray = data.mean(axis=0) if data.ndim > 2 else data
        data_mean = pd.DataFrame(data_mean, columns=[f"Var{i}" for i in range(data.shape[2])])
        return data_mean, data_shape

    def _causality(self, vars):
        test_result = grangercausalitytests(vars, 
                                            maxlag=self.maxlag, 
                                            verbose=False)
        return test_result
    def test(self):
        results = pd.DataFrame(columns=['Cause', 'Effect', 'F Statistic', 'p-value'])
        causality_matrix = np.zeros((self.data_shape[2], self.data_shape[2]))
        p_value_matrix = np.ones((self.data_shape[2], self.data_shape[2]))  # Start with p-values of 1

        for i in range(self.data_shape[2]):
             for j in range(self.data_shape[2]):
                if i != j:
                    test_result = self._causality(self.data[[f"Var{i}", f"Var{j}"]])
                    # Extract the test statistic and p-value from the shortest lag
                    p_value = test_result[1][0]['ssr_chi2test'][1]
                    f_statistic = test_result[1][0]['ssr_chi2test'][0]
                    results = results._append({'Cause': f"Var{i}", 'Effect': f"Var{j}", 
                                              'F Statistic': f_statistic, 'p-value': p_value}, 
                                              ignore_index=True)
                    p_value_matrix[i, j] = p_value
                    if p_value < 0.05:  # If the result is significant
                        causality_matrix[i, j] = f_statistic  # Store the F-statistic

        # Optionally, filter results for significant relationships only (e.g., p-value < 0.05)
        significant_results = results[results['p-value'] < 0.05]

        # Show the results
        # Heatmap visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(causality_matrix, annot=False, cmap='viridis', fmt=".2f")
        plt.title('Granger Causality F-Statistics')
        plt.xlabel('Effect')
        plt.ylabel('Cause')
        plt.show()

        plt.figure(figsize=(10, 8))
        sns.heatmap(p_value_matrix, annot=False, cmap='viridis', fmt=".2f", mask=p_value_matrix >= 0.05)
        plt.title('Granger Causality p-values')
        plt.xlabel('Effect')
        plt.ylabel('Cause')
        plt.show()

        return significant_results

if __name__ == "__main__":
    data_path= "../synthetic-data/data/" + "neural_mass_simulations.npy" #arg_file
    causality = granger_causality(data_path)
    results = causality.test()
