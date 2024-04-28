#%%
import numpy as np
from torch.utils.data import Dataset, DataLoader
# Parameters
class LinearDataset(Dataset):
    def __init__(self, num_points, num_series, window, stride):

        # Initialize the arrays for storing time series
        A = np.random.normal(size=num_points)
        B = np.zeros(num_points)
        C = np.zeros(num_points)
        O = np.zeros(num_points)
        P = np.zeros(num_points)
        D = np.zeros(num_points)
        E = np.zeros(num_points)
        F = np.zeros(num_points)
        G = np.zeros(num_points)
        I = np.zeros(num_points)
        H = np.zeros(num_points)
        M = np.zeros(num_points)
        N = np.zeros(num_points)

        # Generate the series according to the relationships
        for k in range(3, num_points):
            B[k] = 0.7 * A[k-3] + 0.2 * C[k-1] + np.random.normal()
            C[k] = 0.8 * A[k] + 0.5 * C[k] + np.random.normal()
            O[k] = 0.3 * C[k-5] + np.random.normal()
            P[k] = 0.4 * C[k-1] + 0.1 * P[k] + np.random.normal()
            D[k] = 0.3 * B[k-4] + np.random.normal()

        for k in range(2, num_points):
            E[k] = 0.5 * D[k-2] + 0.4 * E[k-2] + np.random.normal()
            F[k] = 0.7 * D[k-2] + np.random.normal()
            M[k] = 0.9 * H[k] + np.random.normal()

        for k in range(num_points):
            G[k] = 0.8 * D[k] + np.random.normal()
            I[k] = 0.2 * F[k] + 0.8 * G[k-1] + np.random.normal()
            H[k] = 0.3 * E[k] + np.random.normal()
            N[k] = 0.7 * H[k-1] + np.random.normal()

        # Now you have 13 series (A to N) each with 4000 points.
        # You can stack them in a 2D array where each row represents a time point and each column a series
        data = np.stack((A, B, C, D, E, F, G, H, I, M, N, O, P), axis=-1)
        num_samples = (num_points - window) // stride + 1
        self.data = np.zeros([num_samples, window,num_series])
        for i in np.arange(num_samples):
            start_x = stride * i
            end_x = start_x + window
            self.data[i,:] = data[start_x:end_x]
        # (num_samples, window, dim)
        self.treatments = self.data[:,:,[0,5]] # A, F (0, 5)
        self.covariates = self.data[:,:,[1,2,3,4,6,7,8]] # B, C, D, E, G, H, I (1,2,3,4,6,7,8)
        self.outcomes = self.data[:,:,[9,10,11,12]]  # M, N, O, P (9,10,11,12)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # batch = dict()
        # batch["prev_treatments"] = self.treatments[idx, :-1, :]
        # batch["current_treatments"] = self.treatments[idx, 1:, :]
        # batch["prev_covariates"] = self.covariates[idx, :-1, :]
        # batch["current_covariates"] = self.covariates[idx, 1:, :]
        # batch["prev_outcomes"] = self.outcomes[idx, :-1, :]
        # batch["current_outcomes"] = self.outcomes[idx, 1:, :]

        batch = {
            "prev_treatments": self.treatments[idx,:-1],
            "curr_treatments": self.treatments[idx,1:],
            "prev_covariates": self.covariates[idx,:-1],
            "curr_covariates": self.covariates[idx,1:],
            "prev_outcomes": self.outcomes[idx,:-1],
            "curr_outcomes": self.outcomes[idx,1:],
        }
        return batch 
if __name__ == "__main__":
    num_points = 4000  # Number of time points
    num_series = 13    # Number of series
    window = 100
    stride = 5
    train_dataset= LinearDataset(num_points, num_series, window, stride)
    train_loader = DataLoader(train_dataset, batch_size = 32)



            



        

        







    # %%
    import matplotlib.pyplot as plt
    import networkx as nx

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes corresponding to each time series
    series = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'M', 'N', 'O', 'P']
    G.add_nodes_from(series)

    # Add edges based on the causal relationships (e.g., B is influenced by A and C)
    G.add_edge('A', 'B', delay=3)
    G.add_edge('C', 'B', delay=1)

    G.add_edge('A', 'C')
    G.add_edge('C', 'C')

    G.add_edge('C', 'O', delay=5)

    G.add_edge('C', 'P', delay=1)
    G.add_edge('P', 'P')

    G.add_edge('B', 'D', delay=4)

    G.add_edge('D', 'E', delay=2)
    G.add_edge('E', 'E', delay=2)

    G.add_edge('D', 'F', delay=2)

    G.add_edge('D', 'G')

    G.add_edge('F', 'I')
    G.add_edge('G', 'I', delay=1)

    G.add_edge('E', 'H')

    G.add_edge('H', 'M')

    G.add_edge('H', 'N', delay=1)

    # Draw the graph
    pos = nx.spring_layout(G, k= 2, iterations=20)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1000, edge_color='black', linewidths=1, font_size=15)

    # Draw edge labels
    edge_labels = {(u, v): d['delay'] for u, v, d in G.edges(data=True) if 'delay' in d}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title('Causal Relationships Between Time Series')
    plt.show()

    # %%
    import matplotlib.pyplot as plt

    # Let's say `data` is your 2D numpy array with each column being a time series
    # If you followed the previous example, `data` has shape (4000, 13) and stores all series

    # Define series names (as per your provided relationships)
    series_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'M', 'N', 'O', 'P']

    # Create subplots - one for each time series
    fig, axs = plt.subplots(len(series_names), 1, figsize=(14, 20))

    # Iterate through each series and create a line plot
    for i, series_name in enumerate(series_names):
        axs[i].plot(data[:200, i], label=series_name)
        axs[i].set_title(series_name)
        axs[i].legend(loc='upper right')
        axs[i].grid(True)

    # Add a title to the entire figure
    fig.suptitle('Simulated Time Series Data', fontsize=16, y=1.02)

    # Layout the plots to prevent overlap
    plt.tight_layout()

    # Show the plots
    plt.show()

# %%
