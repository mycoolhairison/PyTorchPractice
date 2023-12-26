###################################################################
#  This script builds on Karpathy's GPT model, which we previously 
#  trained on the TinyShakespeare dataset.  Here we trained on
#  TinyStories, a collection of children's stories using only words 
#  that young children should know.  See example generated story
#  at the end of the script!
###################################################################
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch_directml
from collections import Counter

class GPT(nn.Module):

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
        self.ffwd = FeedForward(n_embd) #linear relu linear dropout
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
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, bias=False):
        super().__init__()
        self.eps = eps
        self.gamma = torch.ones(dim).to(dml)
        self.beta = torch.zeros(dim).to(dml) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.gamma.shape, self.gamma, self.beta, self.eps)   
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
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)).to(dml))

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
# Data loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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

##################################################
##################################################

# hyperparameters
batch_size = 32
block_size = 256
max_iters = 7500
chunksize = max_iters//50
eval_interval = 750
learning_rate = 5e-5
n_embd = 1024
n_layer = 4
n_head = 16
eval_iters = 200
dropout = .15
dml = torch_directml.device()
cycleTrainingData = True
saveString = f"b{batch_size}b{block_size}e{n_embd}l{n_layer}h{n_head}.pth"
load = True
loadString = "TinyStoriesModel/b32b256e1024l4h16-30000.pth"
generateText = True
contextInit = "A dog walked into a green, dusty room.  The dog's nose started to itch.  Suddenly, there was a loud "
numNewCharsToGenerate = 1000
###

# process data
ttext = open('TinyStoriesModel/stories/TinyStories01.txt', 'r', encoding='utf-8').read()
vtext = open('TinyStoriesModel/stories/TinyStoriesV2-GPT4-valid.txt', 'r', encoding='utf-8').read()

allchars = open('TinyStoriesModel/stories/allchars.txt', 'r', encoding='utf-8').read()
chars = sorted(list(set(allchars)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
val_data = torch.tensor(encode(vtext), dtype=torch.long).to(dml)
train_data = torch.tensor(encode(ttext), dtype=torch.long).to(dml)

m = GPT()
if load:
    m.load_state_dict(torch.load(loadString))
m.to(dml)
for p in m.parameters():
    p.requires_grad = True
m.train()
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(1,max_iters+1):

    ##To avoid keeping all the training data in memory, we cycle through
    ##separate files containing chunks of the original training text.
    if cycleTrainingData:    
        if (iter-1)%chunksize==0 and iter>1:
            chunkNum = (iter//chunksize) + 1
            numStr = str(chunkNum) if chunkNum>=10 else "0"+str(chunkNum)
            fileName = 'TinyStoriesModel/stories/TinyStories' + numStr + '.txt'
            ttext = open(fileName, 'r', encoding='utf-8').read()
            train_data = torch.tensor(encode(ttext), dtype=torch.long).to(dml)
            if chunkNum%10 == 0:
                tss = f"b{batch_size}b{block_size}e{n_embd}l{n_layer}h{n_head}n{numStr}.pth"
                torch.save(m.state_dict(), tss)
            print(f"Loaded chunk {chunkNum}")

    if iter == (max_iters)//2:
        learning_rate = 2e-5
        optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    elif iter == (max_iters*4)//5:
        learning_rate = 5e-6
        optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    elif iter % 100 == 0:
        print(f"step {iter}")
    xb, yb = get_batch('train')
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(m.state_dict(), saveString)

if generateText:
    context = torch.zeros((1,1),dtype=torch.long).to(dml)
    if len(contextInit)>0:
        context = torch.tensor(encode(contextInit), dtype=torch.long).to(dml)
        context = context[None, :]
    print(decode(m.generate(context, max_new_tokens=numNewCharsToGenerate)[0].tolist()))

##############################################################################################################
#  Example generation from the prompt: "Once upon a time, in a warm and
#  sunny place, there was a big pit. A little "
#
#  Once upon a time, in a warm and sunny place, there was a big pit. A little girl was walking near a
#  cafe when she saw a small bird. The bird was not hungry. The girl looked up and asked, "Can you help me?"
#  The bird said, "Yes, I can help you. I am here to help you." The kind lady gave the bird a big hug and
#  they became friends. The girl and the bird ran together, but the ancient little bird was still not a toy.
#  The bird learned to share and treat too. The kind lady and the bird became good friends, and they were
#  good friends. The moral of the story is to nap on your wish every day and wish I could play outside.