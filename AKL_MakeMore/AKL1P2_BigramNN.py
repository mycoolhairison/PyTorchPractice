## Script following Part 2 of Andrej Karpathy's MakeMore Lecture 1.
## Broad goal is to "make more" names given a set of ~30000 names. 
## This script extends Part 1 by incorporating a neural net to the
## bigram generator, using NLL cost function and gradient descent.
## The trained model is the same (i.e. quickly converges to) the
## model in Part 1, but with much more opportunity for improvement!

import torch
import torch.nn.functional as F

## Char to int and int to char dicts, we use '`'==chr(96) as a delimiter
stoi = {chr(c):c-96 for c in range(96,123)}
itos = {stoi[k]:k for k in stoi}

## Create training set from name list
xs, ys = [],[]
words = open('AKL_MakeMore/names.txt', 'r').read().splitlines()
wordString = "`".join(['',*words,''])
for c1,c2 in zip(wordString,wordString[1:]):
    xs.append(stoi[c1])
    ys.append(stoi[c2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
xenc = F.one_hot(xs, num_classes=27).float()

## Initialize weights tensor
W = torch.randn((27, 27), requires_grad=True)

## Training loop
for epoch in range(500):

    #Linear layer
    logits = xenc @ W

    #Softmax
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    #Compute loss (average NLL)
    loss = -probs[torch.arange(len(ys)), ys].log().mean()

    if epoch%10==9:
        print(f"Epoch {epoch+1}: {loss.item():.4f} loss")

    #backward pass
    W.grad = None
    loss.backward()
    W.data += (-50 * W.grad)

## Sampling from the NN model yields the same Cexze output as the Part 1 model.
g = torch.Generator().manual_seed(2147483647)
for i in range(10):
    ix = 0
    s = []
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix==0:
            break
        s.append(itos[ix])
        
    print("".join(s))