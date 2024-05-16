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

device = "cpu"

cache = {}

class AttentionLayer(nn.Module):
    def __init__(self, d_model, d_head, index) -> None:
        super().__init__()
        self.index = index
        self.d_head = d_head
        self.W_Q = nn.Linear(d_model, d_head)
        self.W_K = nn.Linear(d_model, d_head)
        self.W_V = nn.Linear(d_model, d_head)
        self.W_O = nn.Linear(d_head, d_model)
        self.attn_pattern = None
    
    def forward(self, x: Tensor, with_cache):
        """
        x: (batch_size, seq_len, d_model)
        """
        # x = F.scaled_dot_product_attention(
        #     query=Q,
        #     key=K,
        #     value=V,
        #     is_causal=True,
        # )

        Q = self.W_Q.forward(x) # (batch_size, seq_len, d_head)
        K = self.W_K.forward(x) # (batch_size, seq_len, d_head)
        V = self.W_V.forward(x) # (batch_size, seq_len, d_head)

        QK_ = (Q @ K.transpose(-2, -1))
        QK = QK_ / (QK_.shape[-1] ** 0.5)
        large_negative_number = -1e9
        mask = torch.ones_like(QK) * large_negative_number
        mask = torch.triu(mask, diagonal=1)
        attn = F.softmax(QK + mask, dim=-1)
        self.attn_pattern = attn
        if with_cache:
            cache[f'Q_{self.index}'] = Q
            cache[f'K_{self.index}'] = K
            cache[f'V_{self.index}'] = V
            cache[f'attn_{self.index}'] = attn
        x = attn @ V
        return self.W_O(x)


class Model(nn.Module):
    def __init__(self, d_model, d_vocab, d_head) -> None:
        super().__init__()
        self.d_vocab = d_vocab
        self.d_model = d_model
        self.embed = nn.Embedding(d_vocab, d_model)
        self.attn1 = AttentionLayer(d_model, d_head, 1)
        self.attn2 = AttentionLayer(d_model, d_head, 2)
        self.unembed = nn.Linear(d_model, d_vocab)

    def forward(self, input: Tensor, with_cache):
        """
        x: (batch_size, seq_len)
        
        returns: logits (batch_size, seq_len, d_vocab)
        """
        pos = self.positional_encoding(input.shape[1])
        res = self.embed(input) + einops.rearrange(pos, 's d -> () s d')
        res = res + self.attn1(res, with_cache)
        res = res + self.attn2(res, with_cache)
        logits = self.unembed(res)
        return logits

    def forward_train(self, x: Tensor, mask: Tensor):
        """
        x:    (b, seq)
        mask: (b, seq)
        """
        input = x[:,:-1]
        target = x[:,1:] # target = x[:,:-1] # BUG
        mask = mask[:,1:] # TODO: check

        pred_logits = self.forward(input, False)
        pred_log_probs = F.log_softmax(pred_logits, dim=-1)
        target_one_hot = F.one_hot(target, num_classes=self.d_vocab).float()
        loss = (
            einops.reduce(-target_one_hot * pred_log_probs, 'b s v -> b s', 'sum')
            * (~(mask.bool()))
        ).mean()
        return loss

    def positional_encoding(self, seq_len):
        """
        returns: (seq_len, d_model)
        """
        pos = torch.arange(seq_len).float().unsqueeze(-1)
        i = torch.arange(self.d_model // 2).float()
        denom = 1 / (10000 ** (2 * i / self.d_model))
        pe = torch.zeros(seq_len, self.d_model)
        pe[:, 0::2] = torch.sin(pos * denom)
        pe[:, 1::2] = torch.cos(pos * denom)
        return pe.to(device)


#%%


class Trainer:
    def __init__(self, batch_size, seq_len_limit, model: Model):
        self.batch_size = batch_size
        self.seq_len_limit = seq_len_limit
        self.model = model
        self.step = 0

    def train(self, epochs):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        for e in range(epochs):
            optimizer.zero_grad()
            x, mask = self.create_example()
            loss = self.model.forward_train(x, mask)
            loss.backward()
            print(f"Epoch {e}, loss: {loss.item()}")
            optimizer.step()
            # wandb.log({"loss": loss}, self.step)
            attn_pattern_1 = self.model.attn1.attn_pattern[0].detach().numpy()
            attn_pattern_1_as_img = PIL.Image.fromarray(wandb.Image.to_uint8(attn_pattern_1))
            attn_pattern_2 = self.model.attn2.attn_pattern[0].detach().numpy()
            attn_pattern_2_as_img = PIL.Image.fromarray(wandb.Image.to_uint8(attn_pattern_2))
            wandb.log({"attn_pattern_1": wandb.Image(attn_pattern_1_as_img), "attn_pattern_2": wandb.Image(attn_pattern_2_as_img)})
            self.step += 1

    def create_example(self):
        x = []
        mask = []
        for _ in range(self.batch_size):
            s_chunk_len = random.randint(self.seq_len_limit // 4, self.seq_len_limit // 2) # // 2 so that we have at least 2 whole chunks
            seq = []
            l = list(range(self.model.d_vocab))
            random.shuffle(l)
            chunk = l[:s_chunk_len]
            while len(seq) < self.seq_len_limit:
                seq += chunk
            seq = seq[:self.seq_len_limit]
            x.append(seq)
            mask.append([1] * s_chunk_len + [0] * (self.seq_len_limit - s_chunk_len))

        return torch.tensor(x).to(device), torch.tensor(mask).to(device)


d_vocab = 1000
d_model = 512
d_head = 64
model = Model(d_model=d_model, d_vocab=d_vocab, d_head=d_head).to(device)
wandb.init(project="trying-to-find-induction-heads")
trainer = Trainer(batch_size=256, seq_len_limit=128, model=model)
trainer.train(10_000)
wandb.finish()


# %%

# %%
out = model.forward(
    torch.tensor([[100, 200, 300, 400, 500, 600, 500, 500, 500]]).to(device),
    True
)

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
# print(OV.shape)
# # Image.fromarray((QK.detach().cpu().numpy() * 255).astype('uint8')).show()
# imshow(OV.detach().cpu().numpy())
OV.std(), OV.mean(), OV.median(), OV.min(), OV.max()
# %%
QK = model.attn2.W_Q.weight.T @ model.attn2.W_K.weight
imshow(QK.detach().cpu().numpy())
# %%

probs = F.softmax(out, dim=-1).argmax(dim=-1)
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
