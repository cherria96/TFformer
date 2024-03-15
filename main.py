import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


if __name__ == "__main__":
    data_path = '../synthetic-data/data/henon_map_dataset.npy'
    dataset  = np.load(data_path)
    dataset = dataset.squeeze()
    # print(dataset_collection.shape) # (4096,6)
    train_data = dataset[:3600]
    test_data = dataset[3600:]

    train = pd.DataFrame(train_data, columns =["prev_A", "X", "prev_Y", "static_inputs", "curr_A", "active_entries"])
    #print(train)
    ## Data loader 
    train_loader = DataLoader(train, batch_size = 10)
    # test_loader = DataLoader(test, batch_size=10)
