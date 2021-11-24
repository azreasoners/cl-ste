import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms

from sat import CNF
from ste import B, reg_bound, reg_cnf


######################################
# Write the CNF Program if Not Exists
######################################

"""
Write a clause below for r in {0, ..., 1998}
-sum(r) v V_{abc + def = r} predict(a,b,c,d,e,f)
"""

def write_cnf(path_cnf, path_atom2idx):
    # initialize all atoms and assign an index to each of them
    atom2idx = {}
    idx = 1 # the indices start from 1, which follows the dimacs format

    # define 10^6 atoms for predict(n1,n2,n3,n4,n5,n6)
    for n1 in range(10):
        for n2 in range(10):
            for n3 in range(10):
                for n4 in range(10):
                    for n5 in range(10):
                        for n6 in range(10):
                            atom2idx[f'predict({n1},{n2},{n3},{n4},{n5},{n6})'] = idx
                            idx += 1

    # define 1999 atoms for sum
    for r in range(1999):
        atom2idx[f'sum({r})'] = idx
        idx += 1

    # construct the CNF
    numClauses = 0
    cnf2 = ''

    # add 1999 rules of the form -sum(r) v Disjunction_{d1,d2,d3,d4,d5,d6 | d1d2d3+d4d5d6=r} predict(d1,d2,d3,d4,d5,d6)
    for r in range(1999):
        atom = atom2idx[f'sum({r})']
        atoms = [str(atom2idx[f'predict({d1},{d2},{d3},{d4},{d5},{d6})']) \
                    for d1 in range(10) 
                    for d2 in range(10) 
                    for d3 in range(10) 
                    for d4 in range(10)
                    for d5 in range(10)
                    for d6 in range(10)
                    if ((d1+d4)*100+(d2+d5)*10+d3+d6)==r]
        atoms = ' '.join(atoms)
        cnf2 += f'{atoms} -{atom} 0\n'
        numClauses += 1

    cnf1 = f'p cnf {idx-1} {numClauses}\n'
    with open(path_cnf, 'w') as f:
        f.write(cnf1 + cnf2)
    json.dump(atom2idx, open(path_atom2idx,'w'))
    return atom2idx

def read_cnf(path_cnf, path_atom2idx):
    try:
        cnf = CNF(dimacs=path_cnf)
        atom2idx = json.load(open(path_atom2idx))
    except:
        atom2idx = write_cnf(path_cnf, path_atom2idx)
        cnf = CNF(dimacs=path_cnf)
    return cnf.C, atom2idx

######################################
# Define the neural network and loss
######################################

# Fix the randomly initialized model for debugging
def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.manual_seed(64)
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    if isinstance(m, nn.Conv2d):
        torch.manual_seed(64)
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)

class Net(nn.Module):
    def __init__(self, reg, lr, bn, device, b):
        """
        @param reg: a list of strings, each string is in {cnf, bound} denoting a regularizer
            cnf: a string denoting whether L_cnf is used in training
            bound: a string denoting whether L_bound is used in training
        @param lr: a positive float number denoting the learning rate
        """
        super(Net, self).__init__()
        if bn:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 6, 5), # 6 is the output chanel size; 5 is the kernal size; 1 (chanel) 28 28 -> 6 24 24
                nn.BatchNorm2d(6),
                nn.MaxPool2d(2, 2),  # kernal size 2; stride size 2; 6 24 24 -> 6 12 12
                nn.ReLU(True),       # inplace=True means that it will modify the input directly thus save memory
                nn.Conv2d(6, 16, 5), # 6 12 12 -> 16 8 8
                nn.BatchNorm2d(16),
                nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
                nn.ReLU(True) 
            )
            self.classifier =  nn.Sequential(
                nn.Linear(16 * 4 * 4, 120),
                nn.BatchNorm1d(120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.BatchNorm1d(84),
                nn.ReLU(),
                nn.Linear(84, 10)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 6, 5), # 6 is the output chanel size; 5 is the kernal size; 1 (chanel) 28 28 -> 6 24 24
                nn.MaxPool2d(2, 2),  # kernal size 2; stride size 2; 6 24 24 -> 6 12 12
                nn.ReLU(True),       # inplace=True means that it will modify the input directly thus save memory
                nn.Conv2d(6, 16, 5), # 6 12 12 -> 16 8 8
                nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
                nn.ReLU(True) 
            )
            self.classifier =  nn.Sequential(
                nn.Linear(16 * 4 * 4, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10)
            )
        self.cnf, self.bound = [s in reg for s in ('cnf', 'bound')]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.C, self.atom2idx = read_cnf('./cnf/mnistAdd3.cnf', './cnf/mnistAdd3.atom2idx')
        self.numClauses, self.numAtoms = self.C.shape
        self.apply(weight_init)
        self.C = self.C.to(device)

    def forward(self, x):
        """
        @param x: NN input of shape (batch_size, 6, 1, 28, 28)
        """
        # each data instance in a batch has 6 images; we group all images in a batch together
        x = x.view(-1, 1, 28, 28) # (batch_size*6, 1, 28, 28)
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x # (batch_size*6, 10)

    def do_gradient_update(self, data, debug, device, hyper, b):
        """
        @param data: a tuple (images, label) where images is of shape (6, 1, 28, 28) denoting 6 digit images
                                                   and label is an integer in {0, ..., 1999}
        @param debug: a Boolean value denoting whether printing out more info for debugging
        @param hyper: a list of float numbers denoting hyper-parameters
        @param b: a string in {B, Bi, Bs} denoting 3 combinations of b(x) and STE
        """
        images, label = data
        output = self.forward(images.to(device))
        losses = []
        if self.cnf: 
            C, v, g = self.vg_gen(output, label, device)
            losses.append(reg_cnf(C, v, g) * hyper[0])
        if self.bound: 
            losses.append(reg_bound(output) * hyper[1])

        loss = sum(losses)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        if debug:
            print('output[0]')
            print(output[0])

        self.optimizer.zero_grad()
        return loss

    # designed for batchSize = 1
    def vg_gen(self, output, label, device):
        batch_size = label.shape[0]
        g = torch.zeros((batch_size, self.numAtoms), device=device)
        v = torch.zeros((batch_size, self.numAtoms), device=device)

        # 1. encode given information from label into g
        idx = self.atom2idx[f'sum({label[0]})'] - 1
        g[0, idx] = 1

        # 2. construct v from NN output and g
        output = torch.nn.functional.softmax(output, dim=-1)
        # 2.1 the first 10^6 atoms in v represent predict/4
        v[:,:1000000] = B(torch.cartesian_prod(
            output[0,:],
            output[1,:],
            output[2,:],
            output[3,:],
            output[4,:],
            output[5,:]
            ).prod(dim=-1))
        # 2.3 overwrite v with g
        v = g + v * (g==0).int()

        # 3. simplify C by filtering out the unrelated clauses in C
        C = self.C[self.C[:,idx]==-1]
        return C.to(device), v, g

#############################
# Construct the training and testing data
#############################

class MNIST_Addition(Dataset):

    def __init__(self, dataset, examples):
        self.data = list()
        self.dataset = dataset
        with open(examples) as f:
            for line in f:
                line = line.strip().split(' ')
                self.data.append(tuple([int(i) for i in line]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        i1, i2, i3, i4, i5, i6, r = self.data[index]
        return torch.cat((self.dataset[i1][0], self.dataset[i2][0], self.dataset[i3][0], self.dataset[i4][0], self.dataset[i5][0], self.dataset[i6][0]), 0).unsqueeze(1), r

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])
# used for training
trainDataset = MNIST_Addition(torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform), './data/mnistAdd3_train.txt')
# used to check training and testing accuracy
testLoader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, transform=transform), batch_size=1000)
trainLoader = torch.utils.data.DataLoader(Subset(torchvision.datasets.MNIST('./data/', train=True, transform=transform), range(1000)), batch_size=1000)
