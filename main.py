#%%
# Induction circuits

# synthetic dataset
# 2 layer model, attention only

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import einops
import random
import wandb
import PIL.Image
#%%


#%%



# %%

# %%

#%%
#%%

# %%
#%%
mask[:3, :3]
#%%

#%%


out.argmax(dim=-1)

# %%

# %%
imshow(cache['attn_1'].detach().cpu()[0].numpy())
# %%
imshow(cache['attn_2'].detach().cpu()[0].numpy())
# %%
from matplotlib.pyplot import imshow
torch.set_printoptions(precision=4, sci_mode=False)

QK = model.attn1.W_Q.weight.T @ model.attn1.W_K.weight
print(QK)
imshow(QK.detach().cpu().numpy())
# imshow(model.attn1.V1.weight.T.detach().cpu().numpy())
# %%

OV_l1 = model.attn1.W_O.weight @ model.attn1.W_V.weight
OV_l2 = model.attn2.W_O.weight @ model.attn2.W_V.weight

OV = OV_l2 @ OV_l1

OV.std(), OV.mean(), OV.median(), OV.min(), OV.max()
# %%
QK = model.attn2.W_Q.weight.T @ model.attn2.W_K.weight
imshow(QK.detach().cpu().numpy())
# %%

#%%
probs

# %%

out = model.forward(torch.tensor([
    [0,1]
]).to(device))
probs = F.softmax(out, dim=-1)
probs

#%%

# saved1 = model.attn1.O1.weight
# saved2 = model.attn2.O1.weight

#%%

model.attn1.W_O.weight = saved1
model.attn2.W_O.weight = saved2
#%%

model.attn1.W_O.weight = nn.Parameter(torch.zeros_like(model.attn1.W_O.weight))
model.attn2.W_O.weight = nn.Parameter(torch.zeros_like(model.attn2.W_O.weight))

model.attn1.W_O.weight = nn.Parameter(torch.zeros_like(model.attn1.W_O.weight))
model.attn2.W_O.weight = nn.Parameter(torch.zeros_like(model.attn2.W_O.weight))
