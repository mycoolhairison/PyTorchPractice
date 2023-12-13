## Script following Part 1 of Andrej Karpathy's MakeMore Lecture 1.
## Broad goal is to "make more" names given a set of ~30000 names. 
## This script imports and processes names and generates new
## names according to bigram probabilities.  Initial output
## is poor without incorporating any training / loss function.
## (I mean, I wouldn't name my kid Cexze.)

import torch

## Char to int and int to char dicts, we use '`'==chr(96) as a delimiter
stoi = {chr(c):c-96 for c in range(96,123)}
itos = {stoi[k]:k for k in stoi}

## Process names by storing bigram counts in tensor format
words = open('AKL_MakeMore/names.txt', 'r').read().splitlines()
wordString = "`".join(['',*words,''])
## We initialize ones tensor instead of zeros tensor for smoothing.
N = torch.ones((27,27), dtype=torch.int32)
for c1,c2 in zip(wordString,wordString[1:]):
    N[stoi[c1],stoi[c2]]+=1

## Generate a tensor of bigram probabilities.
P = N.float()
P /= P.sum(1, keepdim=True)

## Generate some names according to bigram probabilities.
g = torch.Generator().manual_seed(2147483647)
for i in range(10):
    ix = 0
    s = []
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix==0:
            break
        s.append(itos[ix])
        
    print("".join(s))

## Introducing negative log likelihood with a sample computation
log_likelihood = 0.0
for i in range(27):
    for j in range(27):
        log_likelihood += (torch.log(P[i, j])*N[i,j])
print(f"{log_likelihood=}")
nll = -log_likelihood
print("avg nll =", (nll/N.sum().item()).item())