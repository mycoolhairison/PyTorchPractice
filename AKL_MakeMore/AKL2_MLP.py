## Script following Andrej Karpathy's MakeMore Lecture 2.
## Broad goal is to "make more" names given a set of ~30000 names. 
## In this lecture an MLP model following Bengio, et al. is built,
## with focus on hyperparameter tuning and other ML essentials.
## With batch_size 256, emb_dim 15, a hidden 200-node linear layer,
## 100,000 epochs, block_size 3, learning_rate split .1/.01,
## and improved initialization, the test loss frequently beats AK's
## 'challenge' value of 2.17, sometimes going as low as 2.12, and
## seemingly without overfit.  Some real names are generated!

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

## Char to int and int to char dicts, we use '`'==chr(96) as a delimiter
stoi = {chr(c):c-96 for c in range(96,123)}
itos = {stoi[k]:k for k in stoi}

## Create training / dev / test sets from name list
words = open('AKL_MakeMore/names.txt', 'r').read().splitlines()
random.shuffle(words)
n1 = int(.8*len(words))
n2 = int(.9*len(words))
Xtr, Ytr, Xdev, Ydev, Xte, Yte = [],[],[],[],[],[]
block_size = 3
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

emb_dim = 15
linear_layer_nodes = 200
C = torch.randn((27, emb_dim))
W1 = torch.randn((block_size*emb_dim, linear_layer_nodes))
b1 = torch.randn(linear_layer_nodes)
W2 = torch.randn((linear_layer_nodes,27)) * .01
b2 = torch.randn(27) * 0
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

for epoch in range(100000):

    batch_size = 256

    ix = torch.randint(0, Xtr.shape[0], (batch_size,))

    #forward pass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1,block_size*emb_dim) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    #print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

    if (epoch+1)%5000==0:
        emb = C[Xdev]
        h = torch.tanh(emb.view(-1,block_size*emb_dim) @ W1 + b1)
        logits = h @ W2 + b2
        lossd = F.cross_entropy(logits, Ydev)
        print(f"Epoch {epoch+1}: DevLoss {lossd.item():.4f}, TrLoss {loss.item():.4f}")
         
    for p in parameters:
        p.grad = None
    loss.backward()
    lr = .1 if epoch<50000 else .01
    for p in parameters:
        p.data += -lr * p.grad

#### If emb_dim is 2, can see how the NN learns to spread out the letters in the plane.
#plt.figure(figsize=(8,8))
#plt.scatter(C[:,0].data, C[:,1].data, s=200)
#for i in range(C.shape[0]):
    #plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color="white")
#plt.grid('minor')
#plt.show()

## Generating samples!! Some are real names!!
for _ in range(20):
    out = []
    context = [0]*block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1,-1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim = 1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        if ix==0:
            break
        out.append(itos[ix])
    print("".join(out))

## Finally, compute the test loss!
emb = C[Xte]
h = torch.tanh(emb.view(-1,block_size*emb_dim) @ W1 + b1)
logits = h @ W2 + b2
losst = F.cross_entropy(logits, Yte)
print(f"JUDGMENT TIME: {losst.item():.4f}")