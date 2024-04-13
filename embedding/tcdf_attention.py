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
from embedding_space import EmbeddingSpace
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


checkpoint_path = "../real_weight/cancersim_unroll_3_3_1.pt"
embedding_space = EmbeddingSpace(checkpoint_path)
variable = "treatment"
results = embedding_space.make_heads(variable = variable)
i1, i2 = np.random.randint(embedding_space.num_layers), np.random.randint(embedding_space.num_heads)
W_Q_tmp, W_K_tmp = results[3][i1, i2, :], results[4][i1, i2, :]
attn_score = (embedding_space.emb_inv_A @ (W_Q_tmp @ W_K_tmp.T) @ embedding_space.emb_inv_A.T)
# attn_score += abs(attn_score.min())  # make it above zero
# jth row of attn_score indicates the attention to calculate jth feature (other features -> jth feature)
num_features = 4  # need this value to calculate values to zero out
window_size = 3  # need this value to calculate values to zero out
tot_features = attn_score.shape[0]

# jth row of attn_score indicates the attention to calculate jth feature (other features -> jth feature)
# example order of total features: (A1, B1, C1, A2, B2, C2, A3, B3, C3)
# causality can exist only from past to future (i.e., A1 -> A3, A2 -> C3 not from C3 -> A2 or C2)
# remove from the attention scores where causality can never exists (skip index)

def set_zero(attn_score, window_size, num_features):
    for tdx in range(window_size):
        attn_score[tdx*num_features:(tdx+1)*num_features,tdx*num_features:] = 0
    return attn_score

def index_calculation(scores) -> list:
    avail_indices = torch.nonzero(scores)
    return avail_indices

def convert_to_coor(idx, tot_features=12):
    x, y = divmod(idx, tot_features)
    return [x, y]

attn_score = set_zero(attn_score=attn_score, window_size=window_size, num_features=num_features)
avail_indices = index_calculation(attn_score)
attn_score += abs(attn_score.min()) + 1    # make attentio score larger than one (as TCDF) and above zero
attn_score = set_zero(attn_score=attn_score, window_size=window_size, num_features=num_features)

# scores is attention scores
seed = 42

s = sorted(attn_score.view(-1).cpu().detach().numpy(), reverse=True)
indices = np.argsort(-1 *attn_score.view(-1).cpu().detach().numpy())

tot_len = len(avail_indices)


#attention interpretation to find tau: the threshold that distinguishes potential causes from non-causal time series
if tot_len<=5:
    potentials = []
    for i in indices:
        if attn_score[i]>1.:
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
potential_idx = np.array(list(map(convert_to_coor, potentials)))
print("potential causes: ", potential_idx)

scramble_idx = list(set(potential_idx[:,1]))
print("scramble:", scramble_idx)

validated = copy.deepcopy(scramble_idx)

#Apply PIVM (permutes the values) to check if potential cause is true cause
for idx in scramble_idx:
    random.seed(seed)

    firstloss = 50.0  # temporary value
    
    realloss = 0.0
    testloss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        
        # prev_outputs = batch["prev_outputs"]
        # print(prev_outputs.shape)    # [batch_size, 57, 3]
        # static_features = batch["static_features"]
        # print(static_features.shape)   # [batch_size, 1]
        # active_entries = batch['active_entries']
        # print(active_entries.shape)  # [batch_size, 57, 3]
        # output = batch['outputs'] # [batch_size, 57, 3]
        # print(output.shape)

        # firstloss is loss when model is trained for one epoch
        # realloss is loss when model is fully trained
        realloss += embedding_space.model.training_step(batch, 1, trainer=False).cpu().data.item()   # give 1 to batch_idx cause is not used in the function (no need)
        # print(realloss)

        if variable == "treatment": 
            x_test2 = batch["prev_treatments"].clone().cpu().numpy()  # [batch_size, 57, 12]
            perm = np.arange(x_test2[:,:,idx].shape[1])
            np.random.shuffle(perm)
            x_test2[:,:,idx] = x_test2[:,:,idx][:,perm]
            shuffled_prev_treat = torch.from_numpy(x_test2)

            x_test3 = batch["current_treatments"].clone().cpu().numpy()  # [batch_size, 57, 12]
            perm = np.arange(x_test3[:,:,idx].shape[1])
            np.random.shuffle(perm)
            x_test3[:,:,idx] = x_test3[:,:,idx][:,perm]
            shuffled_cur_treat = torch.from_numpy(x_test3)

            batch['prev_treatments'] = shuffled_prev_treat
            batch['current_treatments'] = shuffled_cur_treat

        testloss += embedding_space.model.training_step(batch, 1, trainer=False).cpu().data.item()   # give 1 to batch_idx cause is not used in the function (no need)
        # print(testloss)
        
        # # permute by batch columns
        # X_test2 = X_train.clone().cpu().numpy()
        # random.shuffle(X_test2[:,idx,:][0])
        # shuffled = torch.from_numpy(X_test2)
        # if cuda:
        #     shuffled=shuffled.cuda()
        # model.eval()
        # output = model(shuffled)
        # testloss = F.mse_loss(output, Y_train)
        # testloss = testloss.cpu().data.item()
        
    diff = firstloss-realloss  # loss gap after training (larger less trained)
    testdiff = firstloss-testloss
    print("diff:", diff)
    print("testdiff:", testdiff)
    significance = 0.8  # use value from TCDF
    if testdiff>(diff*significance): 
        print("removed!")
        validated.remove(idx) 

