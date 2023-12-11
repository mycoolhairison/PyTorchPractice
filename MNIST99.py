## This script trains a simple neural net to achieve ~99.3% accuracy on the MNIST dataset.

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lrs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(3136,512),
            nn.ReLU(),
            nn.Linear(512,16),
            nn.ReLU(),
            nn.Linear(16,10),
        )
    def forward(self, x):
        return self.lrs(x)

def showSampleData(trainingData, sz=8):
    figure = plt.figure(figsize=(sz,sz))
    for i in range(sz*sz):
        sample_idx = torch.randint(len(trainingData), size=(1,)).item()
        img, label = trainingData[sample_idx]
        figure.add_subplot(sz,sz,i+1)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    figure.tight_layout()
    plt.show()

def trainingLoop(dataloader, model, lossFunction, optimizer):
    model.train()
    for (X,y) in dataloader:
        pred = model(X)
        loss = lossFunction(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def testLoop(dataloader, model, lossFunction, show):
    model.eval()
    sz = len(dataloader.dataset)
    numBatches = len(dataloader)
    testLoss, correct = 0,0

    with torch.no_grad():
        for batch, (X,y) in enumerate(dataloader):
            pred = model(X)
            testLoss += lossFunction(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            ##Optional display of test data with guesses
            if batch==0 and show:
                figure = plt.figure(figsize=(8,8))
                p = X.numpy()
                q = pred.argmax(1).numpy()
                for i in range(64):
                    figure.add_subplot(8,8,i+1)
                    plt.title(q[i])
                    plt.axis("off")
                    plt.imshow(p[i].squeeze(), cmap="gray")
                figure.tight_layout()
                plt.show()
    testLoss/=numBatches
    correct/=sz
    print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {testLoss:>8f} \n")

model = NeuralNet()
batchSize = 64

trainingData = datasets.MNIST(root = "data", train = True, download = True, transform=ToTensor())
testData = datasets.MNIST(root = "data", train = False, download=True, transform=ToTensor())
trainDataLoader = DataLoader(trainingData, batch_size=batchSize, shuffle=True)
testDataLoader = DataLoader(testData, batch_size=batchSize, shuffle=False)

#showSampleData(trainingData)

learningRate = .75
epochs = 10
lossFunction = nn.CrossEntropyLoss()
learningCurve = .8

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    learningRate*=learningCurve
    optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)
    trainingLoop(trainDataLoader, model, lossFunction, optimizer)
    show = (t>epochs-1)
    #Displays some test data with guesses when show is True
    testLoop(testDataLoader, model, lossFunction, show)
print("Done!")