
#%%
import sys
sys.path.append("../")
import torch
from torch.nn import functional as F
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
# from tabulate import tabulate
from tqdm import tqdm, trange
from copy import deepcopy
import numpy as np
from collections import Counter
from model.CT_ourmodel import CT

# config 
class EmbeddingSpace:
    def __init__(self, checkpoint_path):
        # emb = model.get_output_embeddings().weight.data.T.detach()
        model = CT.load_from_checkpoint(checkpoint_path)
        self.model = model
        has_vitals = model.has_vitals
        emb_A = model.A_input_transformation.weight.data.detach() # (16,4)
        emb_X = model.X_input_transformation.weight.data.detach() if has_vitals else None 
        emb_Y = model.Y_input_transformation.weight.data.detach() # (16,1)
        emb_V = model.V_input_transformation.weight.data.detach() # (16,1)

        self.emb_inv_A = emb_A.T
        self.emb_inv_X = emb_X.T if has_vitals else None
        self.emb_inv_Y = emb_Y.T
        self.emb_inv_V = emb_V.T

        self.num_layers = model.num_layer
        self.num_heads = model.num_heads # 2
        self.hidden_dim = model.seq_hidden_units #16
        self.head_size = self.hidden_dim // self.num_heads

        # Extract weights
        # K = torch.cat([model.get_parameter(f"transformer.h.{j}.mlp.c_fc.weight").T
        #                            for j in range(num_layers)]).detach()
        # V = torch.cat([model.get_parameter(f"transformer.h.{j}.mlp.c_proj.weight")
        #                            for j in range(num_layers)]).detach() # (hidden_dim, hidden_dim)
        # W_Q, W_K, W_V = torch.cat([model.get_parameter(f"transformer.h.{j}.attn.c_attn.weight") 
        #                            for j in range(num_layers)]).detach().chunk(3, dim=-1)

        self.final_layer = False

        # cross_attention_to (AY)
        # K = torch.cat([(model.transformer_blocks[j].self_attention_t.linear_layers[-1].weight).T
        #                            for j in range(num_layers)]).detach()
        self.K_t = torch.cat([model.transformer_blocks[j].feed_forwards[0].conv1.weight.squeeze()
                                for j in range(self.num_layers)]).detach()
        self.V_t = torch.cat([model.transformer_blocks[j].feed_forwards[0].conv2.weight.squeeze()
                                for j in range(self.num_layers)]).detach()
        # cross_attention_to
        self.W_Q_t = torch.cat([model.transformer_blocks[j].cross_attention_to.linear_layers[0].weight
                                for j in range(self.num_layers)]).detach()
        self.W_K_t = torch.cat([model.transformer_blocks[j].cross_attention_to.linear_layers[1].weight
                                for j in range(self.num_layers)]).detach()
        self.W_V_t = torch.cat([model.transformer_blocks[j].cross_attention_to.linear_layers[2].weight
                                for j in range(self.num_layers)]).detach()
        self.W_O_t = torch.cat([model.transformer_blocks[j].cross_attention_to.final_layer.weight
                                for j in range(self.num_layers)]).detach() if self.final_layer else None

        # cross_attention_ot (YA)
        self.K_o = torch.cat([model.transformer_blocks[j].feed_forwards[1].conv1.weight.squeeze()
                                for j in range(self.num_layers)]).detach()
        self.V_o = torch.cat([model.transformer_blocks[j].feed_forwards[1].conv2.weight.squeeze()
                                for j in range(self.num_layers)]).detach()

        # cross_attention_ot
        self.W_Q_o = torch.cat([model.transformer_blocks[j].cross_attention_ot.linear_layers[0].weight
                                for j in range(self.num_layers)]).detach()
        self.W_K_o = torch.cat([model.transformer_blocks[j].cross_attention_ot.linear_layers[1].weight
                                for j in range(self.num_layers)]).detach()
        self.W_V_o = torch.cat([model.transformer_blocks[j].cross_attention_ot.linear_layers[2].weight
                                for j in range(self.num_layers)]).detach()
        self.W_O_o = torch.cat([model.transformer_blocks[j].cross_attention_ot.final_layer.weight
                                for j in range(self.num_layers)]).detach() if self.final_layer else None
    def make_heads(self, variable):
        """
        Parameters
            variables: treatment / vital / output 
        """
        if variable == "treatment":
            K, V, W_Q, W_K, W_V, W_O = self.K_t, self.V_t, self.W_Q_t, self.W_K_t, self.W_V_t, self.W_O_t
        elif variable == "output":
            K, V, W_Q, W_K, W_V, W_O = self.K_o, self.V_o, self.W_Q_o, self.W_K_o, self.W_V_o, self.W_O_o
        def _make_heads(K, V, W_Q, W_K, W_V, W_O):
            K_heads = K.reshape(self.num_layers, -1, self.hidden_dim)
            V_heads = V.reshape(self.num_layers, -1, self.hidden_dim)
            d_int = K_heads.shape[1]

            W_Q_heads = W_Q.reshape(self.num_layers, self.hidden_dim, self.num_heads, self.head_size).permute(0, 2, 1, 3)
            W_K_heads = W_K.reshape(self.num_layers, self.hidden_dim, self.num_heads, self.head_size).permute(0, 2, 1, 3)
            W_V_heads = W_V.reshape(self.num_layers, self.hidden_dim, self.num_heads, self.head_size).permute(0, 2, 1, 3)
            W_O_heads = W_O.reshape(self.num_layers, self.num_heads, self.head_size, self.hidden_dim) if self.final_layer else None

            return K_heads, V_heads, d_int, W_Q_heads, W_K_heads, W_V_heads, W_O_heads
        return _make_heads(K, V, W_Q, W_K, W_V, W_O)
if __name__=="__main__":
    # W_QK interpretation
    checkpoint_path = "../real_weight/cancersim_unroll_3_3_1.pt"
    embedding_space = EmbeddingSpace(checkpoint_path)
    variable = "treatment"
    results = embedding_space.make_heads(variable = variable)
    i1, i2 = np.random.randint(embedding_space.num_layers), np.random.randint(embedding_space.num_heads)
    W_Q_tmp, W_K_tmp = results[3][i1, i2, :], results[4][i1, i2, :]
    attn_score = (embedding_space.emb_inv_A @ (W_Q_tmp @ W_K_tmp.T) @ embedding_space.emb_inv_A.T)
    print(variable)
    print(attn_score.shape)
    print(attn_score)

    temp = torch.randn((4,4))
    print("=====================")
    print(temp)
    print("=====================")
    print(temp.view(-1))



"""
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
tmp = (emb_inv_A @ (W_V_tmp @ W_O_tmp) @ emb_A)
all_high_pos = approx_topk(tmp, th0=1, verbose=True) 
get_top_entries(tmp, all_high_pos)
#%%
#%%
all_high_pos = approx_topk(tmp2, th0=1, verbose=True) 
get_top_entries(tmp2, all_high_pos)

"""





