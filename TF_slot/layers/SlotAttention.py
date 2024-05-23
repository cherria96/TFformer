
import torch.nn as nn
import torch
import torch.nn.functional as F

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        # b, n, d = inputs.shape
        b , w, d = inputs.shape
        inputs = inputs.reshape(b, w*d)
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(n_s, w, -1)
        sigma = self.slots_sigma.expand(n_s, w, -1)
        slots = torch.normal(mu, sigma) # (n_s, w, d)
        slots = slots.reshape(n_s, -1) # (n_s, wd)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs) # (b, wd)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            
            q = self.to_q(slots) # (n_s, wd)

            dots = torch.einsum('nd,bd->nb', q, k) * self.scale # (n_s, b)
            # dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=0) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True) 

            updates = torch.einsum('bd,nb->nd', v, attn) # (n_s, wd)
            # updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates,
                slots_prev
            )

            # slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots
