import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tabulate import tabulate
from tqdm import tqdm, trange
from copy import deepcopy
import numpy as np
from collections import Counter

# config 
model = None # transformers model
# emb = model.get_output_embeddings().weight.data.T.detach()
emb = None
emb_inv = emb.T

num_layers = model.config.n_layer
num_heads = model.config.n_head
hidden_dim = model.config.n_embd
head_size = hidden_dim // num_heads

# Extract weights
K = torch.cat([model.get_parameter(f"transformer.h.{j}.mlp.c_fc.weight").T
                           for j in range(num_layers)]).detach()
V = torch.cat([model.get_parameter(f"transformer.h.{j}.mlp.c_proj.weight")
                           for j in range(num_layers)]).detach()

W_Q, W_K, W_V = torch.cat([model.get_parameter(f"transformer.h.{j}.attn.c_attn.weight") 
                           for j in range(num_layers)]).detach().chunk(3, dim=-1)
W_O = torch.cat([model.get_parameter(f"transformer.h.{j}.attn.c_proj.weight") 
                           for j in range(num_layers)]).detach()

K_heads = K.reshape(num_layers, -1, hidden_dim)
V_heads = V.reshape(num_layers, -1, hidden_dim)
d_int = K_heads.shape[1]

W_Q_heads = W_Q.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
W_K_heads = W_K.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
W_V_heads = W_V.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
W_O_heads = W_O.reshape(num_layers, num_heads, head_size, hidden_dim)

# Attention weights interpretation
'''
W_vo, W_QK interpretation
embedded transition matrices (E'W_voE) for all heads and examine the top-k pairs of vocabulary items
manually choose a few heads and present the top-k pairs (k = 50)
-> different heads capture different types of relations betwen pairs of vocabulary items including word parts

'''


def approx_topk(mat, min_k=500, max_k=250_000, th0=10, max_iters=10, verbose=False):
    '''
    approximates the top-k elements in a given matrix (mat).
    The function uses a binary search approach to find a threshold value (th) 
    such that the number of elements in the matrix greater than this threshold 
    lies within a specified range (min_k to max_k).
    '''
    _get_actual_k = lambda th, th_max: torch.nonzero((mat > th) & (mat < th_max)).shape[0]
    th_max = np.inf
    left, right = 0, th0 
    while True:
        actual_k = _get_actual_k(right, th_max)
        if verbose:
            print(f"one more iteration. {actual_k}")
        if actual_k <= max_k:
            break
        left, right = right, right * 2
    if min_k <= actual_k <= max_k:
        th = right
    else:
        for _ in range(max_iters):
            mid = (left + right) / 2
            actual_k = _get_actual_k(mid, th_max)
            if verbose:
                print(f"one more iteration. {actual_k}")
            if min_k <= actual_k <= max_k:
                break
            if actual_k > max_k:
                left = mid
            else:
                right = mid
        th = mid
    return torch.nonzero((mat > th) & (mat < th_max)).tolist()

# Replace tokenizer.decode 
def interpret_attention(weights_matrix, positions):
    interpreted_values = []  # Modify this list to store your interpreted values
    for pos in positions:
        # Example: Interpret the attention weights for a specific position
        interpreted_value = weights_matrix[pos[0], pos[1]].item()
        interpreted_values.append(interpreted_value)
    return interpreted_values

# Modify get_top_entries for numerical data
def get_top_entries(tmp, all_high_pos):
    remaining_pos = all_high_pos
    pos_val = tmp[[*zip(*remaining_pos)]]
    # Example: Interpret the attention weights using the interpret_attention function
    interpreted_values = interpret_attention(tmp, remaining_pos)
    # Modify the following code based on your interpretation logic
    # good_cells = [*map(lambda x: (str(x[0].item()), str(x[1].item())), remaining_pos)]
    # good_tokens = list(map(lambda x: Counter(x).most_common(), zip(*good_cells)))
        
    remaining_pos_best = np.array(remaining_pos)[torch.argsort(pos_val)[:50]] #k = 50
    
    # Example: Interpret the attention weights for the top entries
    interpreted_values_best = interpret_attention(tmp, remaining_pos_best)
    
    return interpreted_values_best

# def get_top_entries(tmp, all_high_pos):
#     '''
#     filters and selects high-scoring pairs from a set of positions in a given matrix (tmp). 
#     '''
#     remaining_pos = all_high_pos
#     pos_val = tmp[[*zip(*remaining_pos)]]
#     good_cells = [*map(lambda x: (tokenizer.decode(x[0]), tokenizer.decode(x[1])), remaining_pos)]
#     good_tokens = list(map(lambda x: Counter(x).most_common(), zip(*good_cells)))
#     remaining_pos_best = np.array(remaining_pos)[torch.argsort(pos_val if reverse_list else -pos_val)[:50]]
#     good_cells_best = [*map(lambda x: (tokenizer.decode(x[0]), tokenizer.decode(x[1])), remaining_pos_best)]
#     # good_cells[:100]
#     # list(zip(good_tokens[0], good_tokens[1]))
#     return good_cells_best

i1, i2 = np.random.randint(num_layers), np.random.randint(num_heads)
W_V_tmp, W_O_tmp = W_V_heads[i1, i2, :], W_O_heads[i1, i2]
tmp = (emb_inv @ (W_V_tmp @ W_O_tmp) @ emb)
all_high_pos = approx_topk(tmp, th0=1, verbose=True) 
get_top_entries(tmp, all_high_pos)

i1, i2 = np.random.randint(num_layers), np.random.randint(num_heads)
W_Q_tmp, W_K_tmp = W_Q_heads[i1, i2, :], W_K_heads[i1, i2, :]
tmp2 = (emb_inv @ (W_Q_tmp @ W_K_tmp.T) @ emb_inv.T)
all_high_pos = approx_topk(tmp2, th0=1, verbose=True) 
get_top_entries(tmp, all_high_pos)




