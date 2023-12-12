## Which hyperparameter values work best on MNIST??
## Here we compare models trained on a variety of values.

import torch
from torch import nn
from datetime import datetime
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, linearLayerParameters):
        super().__init__()
        ##The model has two convolutional layers with a variable number of linear layers.
        layers = [
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
            ]
        llps = [3136]
        llps.extend(linearLayerParameters)
        llps.append(10)
        numLinearLayers = len(llps)-1
        for i in range(numLinearLayers):
            layers.append(nn.Linear(llps[i],llps[i+1]))
            if i<numLinearLayers-1:
                layers.append(nn.ReLU())
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

def trainingLoop(dataloader, model, lossFunction, optimizer):
    model.train()
    for (X,y) in dataloader:
        pred = model(X)
        loss = lossFunction(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def testLoop(dataloader, model, lossFunction, t) -> float:
    model.eval()
    sz = len(dataloader.dataset)
    numBatches = len(dataloader)
    testLoss, correct = 0,0
    with torch.no_grad():
        for (X,y) in dataloader:
            pred = model(X)
            testLoss += lossFunction(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    testLoss/=numBatches
    correct/=sz
    file = open("hyperoutput.txt", "a")
    print(f"E{t+1}: {(100*correct):>0.2f}%")
    file.write(f"E{t+1}: {(100*correct):>0.2f}%\n")
    file.close()
    return correct

trainingData = datasets.MNIST(root = "data", train = True, download = True, transform=ToTensor())
testData = datasets.MNIST(root = "data", train = False, download=True, transform=ToTensor())
lossFunction = nn.CrossEntropyLoss()
epochs = 100

file = open("hyperoutput.txt", "w")
file.close()

hyperparameterOptions = [
    [[512,16],64,.5,.8], [[512,16],64,.2,.8], [[512,16],64,.4,.9], [[512,16],64,.3,.9], [[512,16],16,.4,.9],
    [[512,16],16,.3,.9], [[512,16],64,.5,.5], [[512,16],64,.2,.5], [[512,16],16,.5,.8], [[512,16],16,.2,.8],
    [[512,16],256,.5,.8], [[512,16],256,.2,.8], [[512,16],64,.01,1], [[512,16],16,.01,1], [[1024,512,16],64,.5,.8],
    [[512,512,16],64,.5,.8], [[512,16,16],64,.5,.8], [[2048,16],64,.5,.8], [[1024,256,16],64,.5,.8],
    [[1024],64,.5,.8], [[512],64,.5,.8], [[128],64,.5,.8], [[64],64,.5,.8], [[16],64,.5,.8],
    [[1024],64,.4,.9], [[512],64,.4,.9], [[128],64,.4,.9], [[64],64,.4,.9], [[16],64,.4,.9],
    [[1024],20,.5,.8], [[512],20,.5,.8], [[128],20,.5,.8], [[64],20,.5,.8], [[16],20,.5,.8],
    [[16,16,16],64,.6,.8], [[16,16,16,16],64,.6,.8], [[64,64],64,.6,.8], [[64,32],64,.6,.8]
]

# batchSizes = [64,16,256]
# learningRates = [.7,.5]
# learningDecays = [.8,.6]
# linearLayerList = [[512,16],[1024,512,16],[512,16,16],[2048,16],[1024,256,16],[1024],[128],[64],[16,16,16]]


for linearLayer,batchSize,learningRate,learningDecay in hyperparameterOptions:
    model = NeuralNet(linearLayer)
    trainDataLoader = DataLoader(trainingData, batch_size=batchSize, shuffle=True)
    testDataLoader = DataLoader(testData, batch_size=batchSize, shuffle=False)
    
    file = open("hyperoutput.txt", "a")
    file.write(f"Training with linear layers {linearLayer}, batch size {batchSize}, learning rate/decay {learningRate}/{learningDecay}\n")
    print(f"Training with linear layers {linearLayer}, batch size {batchSize}, learning rate/decay {learningRate}/{learningDecay}")
    file.close()

    startTime = datetime.now()
    accuracies = [.1]
    
    for t in range(epochs):
        learningRate*=learningDecay
        optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)
        trainingLoop(trainDataLoader, model, lossFunction, optimizer)
        accuracy = testLoop(testDataLoader, model, lossFunction, t)
        accuracies.append(accuracy)
        ## Concludes if first epoch lands in a bad valley, or if model is not impressive after 4 epochs,
        ## or after 4 consecutive similar accuracies.
        if accuracy<.2 or (t>2 and (accuracy<.978 or (max(accuracies[-4:])-min(accuracies[-4:])<.0008))):
            file = open("hyperoutput.txt", "a")
            if accuracy>.993:
                file.write("-----------------------------------LOOK AT ME-----------------------------------\n")
            file.write(f"Achieved {(100*accuracy):>0.2f}% accuracy after {t+1} epochs in time {str(datetime.now()-startTime)[2:7]}\n")
            print(f"Achieved {(100*accuracy):>0.2f}% accuracy after {t+1} epochs in time {str(datetime.now()-startTime)[2:7]}")
            if accuracy>.993:
                file.write("-----------------------------------LOOK AT ME-----------------------------------\n\n")
            else:
                file.write("\n")
            file.close()
            break