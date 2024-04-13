import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
import pandas as pd
import numpy as np
import heapq
import copy
import os
import sys
from torch.utils.data import DataLoader
sys.path.append("../")
from model.CT_ourmodel import CT
from torch.nn import functional as F
from model.utils import unroll_temporal_data
from src.data.cancer_sim.dataset import SyntheticCancerDatasetCollection

torch.set_default_dtype(torch.float64)

# cancer_sim 
num_patients = {'train': 10000, 'val': 10000, 'test': 1000}
datasetcollection = SyntheticCancerDatasetCollection(chemo_coeff = 3.0, radio_coeff = 3.0, num_patients = num_patients, window_size =15, 
                                    max_seq_length = 60, projection_horizon = 5, 
                                    seed = 42, lag = 0, cf_seq_mode = 'sliding_treatment', treatment_mode = 'multiclass')
datasetcollection.process_data_multi()

# Example of iterating over the DataLoader
config = {
    "lr" : 0.01,
    "epochs" : 150,
    "batch_size": 256,
    "fc_hidden_units": 32,
    "has_vital": False
}
keys = ['prev_treatments', 'current_treatments', 'current_covariates', 'outputs', 'active_entries', 'unscaled_outputs', 'prev_outputs']
if config["has_vital"]:
    keys.append(['vitals', 'next_vitals'])
for key in keys:
    observed_nodes_list= list(range(datasetcollection.train_f.data[key].shape[-1]))
    datasetcollection.train_f.data[key],_,_ = unroll_temporal_data(datasetcollection.train_f.data[key], observed_nodes_list, window_len = 3)
    dim_X = 0 
    if key == 'prev_treatments':
        dim_A = datasetcollection.train_f.data[key].shape[-1] # Dimension of treatments
    elif key == 'vitals':
        dim_X = datasetcollection.train_f.data[key].shape[-1] # Dimension of vitals
    elif key == "outputs":
        dim_Y = datasetcollection.train_f.data[key].shape[-1] # Dimension of outputs
    elif key == "current_covariates":
        dim_V = datasetcollection.train_f.data[key].shape[-1] # Dimension of static inputs
for key in keys:
    observed_nodes_list= list(range(datasetcollection.val_f.data[key].shape[-1]))
    datasetcollection.val_f.data[key],_,_ = unroll_temporal_data(datasetcollection.val_f.data[key], observed_nodes_list, window_len = 3)
    dim_X = 0 
    if key == 'prev_treatments':
        dim_A = datasetcollection.val_f.data[key].shape[-1] # Dimension of treatments
    elif key == 'vitals':
        dim_X = datasetcollection.val_f.data[key].shape[-1] # Dimension of vitals
    elif key == "outputs":
        dim_Y = datasetcollection.val_f.data[key].shape[-1] # Dimension of outputs
    elif key == "current_covariates":
        dim_V = datasetcollection.val_f.data[key].shape[-1] # Dimension of static inputs
    
    
# dim_A = 4  # Dimension of treatments
# dim_X = 0  # Dimension of vitals
# dim_Y = 1  # Dimension of outputs
# dim_V = 1  # Dimension of static inputs

batch_size = config['batch_size']
epoch = config['epochs']
fc_hidden_units = config['fc_hidden_units']
train_loader = DataLoader(datasetcollection.train_f, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(datasetcollection.val_f, batch_size=batch_size, shuffle=False)


checkpoint_path = "../real_weight/cancersim_5_256_150.pt"
model = CT.load_from_checkpoint(checkpoint_path)

# scores is attention scores
seed = 42
num_features = 3
window_size = 3
tot_features = num_features*window_size
scores = torch.normal(1, 1, size=(tot_features, tot_features))
scores += abs(scores.min()) + 0.0001
# example order of total features: (A1, B1, C1, A2, B2, C2, A3, B3, C3)
# causality can exist only from past to future (i.e., A1 -> A3, A2 -> C3 not from C3 -> A2 or C2)
# remove from the attention scores where causality can never exists (skip index)

def skip_index_calculation(window_size, num_features) -> list:

    skip_indices = []
    # TODO
    return skip_indices

skip_indices = skip_index_calculation(window_size=window_size, num_features=num_features)
s = sorted(scores.view(-1).cpu().detach().numpy(), reverse=True)
indices = np.argsort(-1 *scores.view(-1).cpu().detach().numpy())

tot_len = len(s) - len(skip_indices)

#attention interpretation to find tau: the threshold that distinguishes potential causes from non-causal time series
if len(s)<=5:
    potentials = []
    for i in indices:
        if i in skip_indices:
            continue
        if scores[i]>1.:
            potentials.append(i)
else:
    potentials = []
    gaps = []
    for i in range(len(s)-1):
        if s[i]<1.: #tau should be greater or equal to 1, so only consider scores >= 1
            break
        gap = s[i]-s[i+1]
        gaps.append(gap)
    sortgaps = sorted(gaps, reverse=True)

    for i in range(0, len(gaps)):
        largestgap = sortgaps[i]
        index = gaps.index(largestgap)
        ind = -1
        if index<((tot_len-1)/2): #gap should be in first half
            if index>0:
                ind=index #gap should have index > 0, except if second score <1
                break
    if ind<0:
        ind = 0
            
    potentials = indices[:ind+1].tolist()

print("Potential causes: ", potentials)   # this cause will be only on selected features...
validated = copy.deepcopy(potentials)

#Apply PIVM (permutes the values) to check if potential cause is true cause
for idx in potentials:
    random.seed(seed)
    for batch_idx, batch in enumerate(train_loader):
        
        prev_treatments = batch["prev_treatments"]
        print(prev_treatments.shape)
        vitals = None # batch["vitals"]
        prev_outputs = batch["prev_outputs"]
        print(prev_outputs.shape)
        static_features = batch["static_features"]
        print(static_features.shape)
        current_treatments = batch["current_treatments"]
        print(current_treatments.shape)
        active_entries = batch['active_entries']
        print(active_entries.shape)

        # permute by batch columns
        break

#     X_test2 = X_train.clone().cpu().numpy()
#     random.shuffle(X_test2[:,idx,:][0])
#     shuffled = torch.from_numpy(X_test2)
#     if cuda:
#         shuffled=shuffled.cuda()
#     model.eval()
#     output = model(shuffled)
#     testloss = F.mse_loss(output, Y_train)
#     testloss = testloss.cpu().data.item()
    
#     diff = firstloss-realloss
#     testdiff = firstloss-testloss

#     if testdiff>(diff*significance): 
#         validated.remove(idx) 


# weights = []

# #Discover time delay between cause and effect by interpreting kernel weights
# for layer in range(layers):
#     weight = model.dwn.network[layer].net[0].weight.abs().view(model.dwn.network[layer].net[0].weight.size()[0], model.dwn.network[layer].net[0].weight.size()[2])
#     weights.append(weight)

# causeswithdelay = dict()    
# for v in validated: 
#     totaldelay=0    
#     for k in range(len(weights)):
#         w=weights[k]
#         row = w[v]
#         twolargest = heapq.nlargest(2, row)
#         m = twolargest[0]
#         m2 = twolargest[1]
#         if m > m2:
#             index_max = len(row) - 1 - max(range(len(row)), key=row.__getitem__)
#         else:
#             #take first filter
#             index_max=0
#         delay = index_max *(dilation_c**k)
#         totaldelay+=delay
#     if targetidx != v:
#         causeswithdelay[(targetidx, v)]=totaldelay
#     else:
#         causeswithdelay[(targetidx, v)]=totaldelay+1
# print("Validated causes: ", validated)
