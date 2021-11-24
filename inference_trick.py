import argparse
import csv
import importlib
import glob
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
torch.cuda.empty_cache()

#############################
# Palm Sudoku dataset
#############################

def sudoku_str2tensor(s):
    return torch.tensor([int(i) for i in s], dtype=torch.long)

def read_csv(dataset_path):
    print(f'Reading Sudoku data from {dataset_path}')
    with open(dataset_path) as f:
        reader = csv.reader(f, delimiter=',')
        return [(sudoku_str2tensor(q).view(1,9,9).float(), sudoku_str2tensor(a)-1) for q, a in reader]

class Sudoku_dataset_palm(Dataset):
    """
    Note that the Palm train/test/val dataset contain 180k, 18k, 18k puzzles 
    with a uniform distribution of givens between 17 and 34 (which are 18 kinds)
    """
    def __init__(self, dataset_path, given=None):
        self.data = read_csv(dataset_path)
        if given:
            self.data = [(inp, label) for (inp, label) in self.data if ((inp != 0).sum().item() in given)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#############################
# Park CNN for inference trick
#############################

class Sudoku_CNN(nn.Module):
    def __init__(self, bn):
        """
        @param bn: a boolean value denoting whether batch normalization is used

        Remark:
            The same CNN from [Park 2018] for solving Sudoku puzzles given as 9*9 matrices
            https://github.com/Kyubyong/sudoku
        """
        super(Sudoku_CNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, 512, kernel_size=3, stride=1, padding=1))
        if bn: layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU())
        for i in range(8):
            layers.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
            if bn: layers.append(nn.BatchNorm2d(512))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(512, 9, kernel_size=1))
        if bn: layers.append(nn.BatchNorm2d(9))
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        """
        @param x: NN input of shape (batch_size, 1, 9, 9)
        """
        x = self.nn(x) # (batch_size, 9, 9, 9)
        x = x.permute(0,2,3,1)
        x = x.view(-1,81,9)
        return x # (batch_size, 81, 9)

def trick_inference(model, X):
    """
    @param model: a Pytorch model
    @param X: a tensor of shape (batchSize, 1, 9, 9), denoting the input to the NN
    """
    model.eval()
    batchSize = X.shape[0]
    pred = X.clone().view(batchSize, 81) # values {0,1,...,9} of shape (batchSize, 81)
    while 0 in pred:
        output = model(pred.view(batchSize,1,9,9)).view(batchSize,81,9) # (batchSize, 81, 9)
        output = torch.nn.functional.softmax(output, dim=-1) # (batchSize, 81, 9)
        values, indices = output.max(dim=-1) # (batchSize,81), (batchSize,81)
        values[pred != 0] = 0.
        cellIndices = values.argmax(dim=-1) # (batchSize)
        for batchIdx, cellIdx in enumerate(cellIndices.tolist()):
            if pred[batchIdx,cellIdx] == 0:
                pred[batchIdx,cellIdx] = indices[batchIdx,cellIdx] + 1 # pred contains number 0-9, where 1-9 are labels
    return pred - 1

def main(args):
    start_time = time.time()
    # set the device
    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')
    print(f'device is {device.type}')
    # set the testing dataloader
    if 'Palm' in args.domain:
        testDataset = Sudoku_dataset_palm('./data/palm_sudoku/test.csv', given=args.given)
        if args.given:
            print(f'Test on the Palm test data with {str(args.given)} given numbers')
        else:
            print('Test on the complete Palm test data with 17~34 given numbers')
        test_loader = torch.utils.data.DataLoader(testDataset, batch_size=32, shuffle=False)
    else:
        domain = importlib.import_module(f'problems.{args.domain}')
        test_loader = domain.testLoader

    if args.file == 'NA':
        # obtain all .pt files
        filenames = sorted(glob.glob(f'{args.folder}/*.pt'), key=os.path.getmtime)
    else:
        filenames = [f'{args.folder}/{args.file}.pt']

    # evaluate trained models on the test dataset 
    # for filename in filenames:
    for filename in reversed(filenames):

        # # Only used if you want to filter some of the models
        # name = 'sudoku_epoch_'
        # if name not in filename:
        #     continue
        # check = filename
        # check = check.split(name)[1].replace('.pt', '')
        # if int(check) < 25:
        #     continue

        model = Sudoku_CNN(bn=args.bn)
        # breakpoint()
        model.load_state_dict(torch.load(filename, map_location=device))
        model.eval()
        model.to(device)
        str1='model: {0}, '.format(filename)

        ##########################
        # accuracy on test data
        ##########################

        # check if total prediction is correct
        correct = 0
        total = 0
        # check if each single prediction is correct
        singleCorrect = 0
        singleTotal = 0

        for _, (X, labels) in enumerate(tqdm(test_loader)):
            batchSize = X.shape[0]
            pred = trick_inference(model, X.to(device))
            correctionMatrix = (labels.int().view(-1,81).to(device) == pred.int())
            correct += correctionMatrix.all(1).sum().item()
            singleCorrect += correctionMatrix.sum().item()
            total += batchSize
            singleTotal += batchSize * 81

        str2='\t{:0.4f} '.format(round(singleCorrect/singleTotal,4))
        str3='\t{:0.4f}'.format(round(correct/total,4))
        print(str1+str2+str3)
    print('--- total time from beginning: %s seconds ---' % int(time.time() - start_time) )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='./models', help='the path to all the trained models')
    parser.add_argument('--file', type=str, default='NA', help='if not NA, only check the acc of model [folder]/[file].pt')
    parser.add_argument('--domain', type=str, default='sudokucnf', help='the file name where we look for testLoader')
    parser.add_argument('--given', default=[], nargs='+', type=int, help='specify the given number of digits in each board')
    parser.add_argument('--gpu', default=False, action='store_true', help='try to use GPU if cuda is available')
    parser.add_argument('--bn', default=False, action='store_true', help='use batch normalization in the model')
    args = parser.parse_args()
    main(args)
