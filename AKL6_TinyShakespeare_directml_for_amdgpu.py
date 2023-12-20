###################################################################
#  Script following Andrej Karpathy's Build GPT Lecture.  Here we
#  implement nanoGPT from scratch and train it on TinyShakespeare.
#  This script builds on the MakeMore series which introduced many
#  layer types; major improvement here is incorporating attention!
#  This variation uses directml for running on Windows with AMD GPU.
#  Sample output at the end of the code.
###################################################################

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch_directml

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 3000
eval_interval = 1500
learning_rate = 1e-3
n_embd = 384
n_layer = 6
n_head = 6
eval_iters = 200
dropout = .2
dml = torch_directml.device()
###

# process data
text = open('tshake.txt', 'r', encoding='utf-8').read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
###

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        #print(f"forward step: x: {idx.get_device()}, y: {targets.get_device() if targets is not None else None}")
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb =self.position_embedding_table(torch.arange(T).to(dml))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            # Reshape logits and targets based on sizes cross_entropy expects
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):

        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)
        xhat = (x-xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        # attention scores
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


# Train and test splits
splitSpot = int(0.9*len(data))
train_data = data[:splitSpot]
val_data = data[splitSpot:]

# Data loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(dml)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(dml)
    #print(f"Get batch: x: {x.get_device()}, y: {y.get_device()}")
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters).to(dml)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = m(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


m = BigramLanguageModel()
m.to(dml)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(1,max_iters+1):
    if iter==(max_iters*3)//5:
        learning_rate = 3e-4
        optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    elif iter % 25 == 0:
        print(f"step {iter}")
    xb, yb = get_batch('train')
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1),dtype=torch.long).to(dml)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

## Generated output!
'''
What ever your breath you have not by old.

GLOUCESTER:
Yet where's day: I'll say your teward, my dear of great.
What news? brother? to hover my subful stands?

QUEEN ELIZABETH:
Wither shouts fall her answer'd heir snarless,
And rades to me no store, and for all I inquest:
This subject hath the head out of that contrary.

QUEEN ELIZABETH:
'Yea, brother with the rotten down; though you art poing,
In to seeming boys a grave too: bark if you
hear wault burthen; and power he is back'd become:
You, my tribune isade.
Sta, no.

GLOUCESTER:
When were what they speak a hand, if it far as he
here, both Marcius?

GLOUCESTER:
Well, do he, not do peace.

CLARENCE:
What's he?
Good motioning, that for condemn'd with thee boldy to-bors
And pray'd them with us that o'closed it,
Great up heavy that I did the crottake:
A drop house And here for my majesty.
A patience, did me in strift: do you must fall
Swear do hold at false ass, you back again,
Whipp'd you, in blame, being, pardon me! God gentlemen,
Were Nedlayer main from suck'd ubled prison's face;
Here is he to our soldiers, and shall in us,
Colse beauthful warm'd before her are forged,
When her find that he did departed coming.
Thou put out what said is time, you'ld good my heart.
'''