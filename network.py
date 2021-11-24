import sys
import torch
import torch.nn as nn
from tqdm import tqdm

# the function to train a neural network model using DiODE
def train(model, trainDataset, epochs: int, save: str, folder: str,
          load: str, checkAcc: bool, testLoader, trainLoader, device,
          debug: bool, plot: bool, bar: bool=False, hyper=None, b='B'):
    # set up training mode
    model.train()
    model.to(device)
    count = 0
    if load != 'NA':
        model.load_state_dict(torch.load(f'{folder}/{load}.pt', map_location=device))
    for epoch in range(epochs):
        total_loss = 0
        iterator = enumerate(tqdm(trainDataset)) if bar else enumerate(trainDataset)
        for _, data in iterator:
            loss = model.do_gradient_update(data, debug, device, hyper, b)
            total_loss += loss.item()
            if plot:
                count += 1
                if count % plot  == 0:
                    # check testing accuracy
                    accuracy, singleAccuracy = testNN(model=model, testLoader=testLoader, device=device)
                    # check training accuracy
                    accuracyTrain, singleAccuracyTrain = testNN(model=model, testLoader=trainLoader, device=device)
                    print(f'        \t{accuracyTrain:0.0f}\t{accuracy:0.0f}')
        # at the end of each epoch, if save is not 'NA', we save the trained model
        if save != 'NA':
            saveModelPath = f'{folder}/{save}_epoch_{epoch+1}.pt'
            print('Storing the trained model into {}'.format(saveModelPath))
            torch.save(model.state_dict(), saveModelPath)

        # if we want to check accuracy for every epoch AND accuracy is not checked for every 10 iterations AND the current epoch is not the last one
        if checkAcc and not plot and epoch + 1 < epochs:
            # check testing accuracy
            accuracy, singleAccuracy = testNN(model=model, testLoader=testLoader, device=device)
            # check training accuracy
            accuracyTrain, singleAccuracyTrain = testNN(model=model, testLoader=trainLoader, device=device)
            # print(f'{singleAccuracyTrain:0.2f}\t{accuracyTrain:0.2f}\t{singleAccuracy:0.2f}\t{accuracy:0.2f}')
            print(f'        \t{accuracyTrain:0.2f}\t{accuracy:0.2f}')

# the function to test a neural network model using a test data loader
def testNN(model, testLoader, device):
    """
    Return a real number "accuracy" in [0,100] which counts 1 for each data instance; 
           a real number "singleAccuracy" in [0,100] which counts 1 for each number in the label 
    @param model: a PyTorch model whose accuracy is to be checked 
    @oaram testLoader: a PyTorch dataLoader object, including (input, output) pairs for model
    """
    # set up testing mode
    model.eval()

    # check if total prediction is correct
    correct = 0
    total = 0
    # check if each single prediction is correct
    singleCorrect = 0
    singleTotal = 0
    with torch.no_grad():
        for data, target in testLoader:
            output = model(data.to(device))
            if target.shape == output.shape[:-1]:
                pred = output.argmax(dim=-1) # get the index of the max value
            elif target.shape == output.shape:
                pred = (output >= 0).int()
            else:
                print(f'Error: none considered case for output with shape {output.shape} v.s. label with shape {target.shape}')
                sys.exit()
            target = target.to(device).view_as(pred)
            correctionMatrix = (target.int() == pred.int()).view(target.shape[0], -1)
            correct += correctionMatrix.all(1).sum().item()
            total += target.shape[0]
            singleCorrect += correctionMatrix.sum().item()
            singleTotal += target.numel()
    accuracy = 100. * correct / total
    singleAccuracy = 100. * singleCorrect / singleTotal
    return accuracy, singleAccuracy

# the class of a general fully-connect neural network (also called Multi-Layer Perception)
class FC(nn.Module):
    def __init__(self, *sizes):
        super(FC, self).__init__()
        layers = []
        print('Neural Network (MLP) Structure: {}'.format(sizes))
        for i in range(len(sizes)-2):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)