###################################################################
#  Script following Andrej Karpathy's MakeMore Lecture 5.
#  Broad goal is to "make more" names given a set of ~30000 names. 
#  The goal of Lecture 5 is to incorporate newer techniques,
#  such as a convolutional layer, and also to clean up / torchify
#  a lot of the code from previous lectures.  With some minor
#  tweaking of parameters, we obtain a test loss of 1.982, beating
#  the challenge value of 1.99.  Many real names are generated!
###################################################################

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in,fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum=momentum
        self.training=True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):

        if self.training:
            # Note that the dimension locations here differ from the torch batchnorm imp.
            dim = 0 if x.ndim == 2 else (0,1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x-xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1- self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []
    
class Embedding:
    def __init__(self, num_embeddings, emb_dim):
        self.weight = torch.randn((num_embeddings, emb_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]
    
class FlattenConsecutive:

    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1]==1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    
    def parameters(self):
        return []
    
class Sequential:

    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    
## Hyperparameters
emb_dim = 24
num_hidden_nodes = 128
block_size = 8


## Char to int and int to char dicts, we use '`'==chr(96) as a delimiter
stoi = {chr(c):c-96 for c in range(96,123)}
itos = {stoi[k]:k for k in stoi}

## Create training / dev / test sets from name list
words = open('AKL_MakeMore/names.txt', 'r').read().splitlines()
random.shuffle(words)
n1 = int(.8*len(words))
n2 = int(.9*len(words))
Xtr, Ytr, Xdev, Ydev, Xte, Yte = [],[],[],[],[],[]
for i,w in enumerate(words):
    context = [0]*block_size
    for ch in w + '`':
        ix = stoi[ch]
        if i<n1:
            Xtr.append(context)
            Ytr.append(ix)
        elif i<n2:
            Xdev.append(context)
            Ydev.append(ix)
        else:
            Xte.append(context)
            Yte.append(ix)
        context = context[1:] + [ix]
Xtr = torch.tensor(Xtr)
Ytr = torch.tensor(Ytr)
Xdev = torch.tensor(Xdev)
Ydev = torch.tensor(Ydev)
Xte = torch.tensor(Xte)
Yte = torch.tensor(Yte)

## This should probably be obtained from the dataset instead of hardcoded,
## but here we're guaranteed lowercase english + our own special char `
vocab_size = 27

model = Sequential([
    Embedding(vocab_size, emb_dim),
    FlattenConsecutive(2),
    Linear(emb_dim*2, num_hidden_nodes, bias = False),
    BatchNorm1d(num_hidden_nodes),
    Tanh(),
    FlattenConsecutive(2),
    Linear(num_hidden_nodes*2, num_hidden_nodes, bias = False),
    BatchNorm1d(num_hidden_nodes),
    Tanh(),
    FlattenConsecutive(2),
    Linear(num_hidden_nodes*2, num_hidden_nodes, bias = False),
    BatchNorm1d(num_hidden_nodes),
    Tanh(),
    Linear(num_hidden_nodes, vocab_size),
])

# with torch.no_grad():
#     model.layers[-1].weight *= .1
#     for layer in model.layers[:-1]:
#         if isinstance(layer, Linear):
#             layer.weight *= 5/3

parameters = model.parameters()
for p in parameters:
    p.requires_grad = True

batch_size = 64
max_steps = 70000
lossi = []

for epoch in range(max_steps):

    #create minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    #forward pass
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)
    #print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    lr = .1 if epoch<40000 else .01
    for p in parameters:
        p.data += -lr * p.grad

    if (epoch+1)%10000==0:
        logitsd = model(Xdev)
        lossd = F.cross_entropy(logitsd, Ydev)
        print(f"Epoch {epoch+1}: DevLoss {lossd.item():.4f}, TrLoss {loss.item():.4f}")
    lossi.append(loss.log10().item())


@torch.no_grad()
def split_loss(split):
    x,y = {
        'train': (Xtr,Ytr),
        'dev': (Xdev,Ydev),
        'test': (Xte,Yte),
    }[split]
    logitst = model(x)
    loss = F.cross_entropy(logitst, y)
    print(split, loss.item())

for layer in model.layers:
    layer.training = False

split_loss('train')
split_loss('dev')
split_loss('test')

## Generating samples!!
for _ in range(15):
    out = []
    context = [0]*block_size
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim = 1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        if ix==0:
            break
        out.append(itos[ix])
    print("".join(out))