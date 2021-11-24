import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms

from sat import CNF
from ste import B, Bi, Bs, reg_bound, reg_cnf


######################################
# Write the CNF Program if Not Exists
######################################

"""
CNF for b_p(x)+iSTE
Write 6 clauses below for all d in {0,...,9}
-in(d, true) v digit(0, d) v digit(1, d) v digit(2, d) v digit(3, d) v digit(4, d)
-in(d, false) v -digit(0, d)
-in(d, false) v -digit(1, d)
-in(d, false) v -digit(2, d)
-in(d, false) v -digit(3, d)
-in(d, false) v -digit(4, d)
"""

def write_cnf(path_cnf, path_atom2idx):
    # initialize all atoms and assign an index to each of them
    atom2idx = {}
    idx = 1 # the indices start from 1, which follows the dimacs format
    # define 5*10 atoms for digit/2
    for i in range(5):
        for d in range(10):
            atom2idx[f'digit({i},{d})'] = idx
            idx += 1
    # define 10*2 atoms for in/2
    for d in range(10):
        for b in range(2):
            atom2idx[f'in({d},{b})'] = idx
            idx += 1

    # construct the CNF
    numClauses = 0
    cnf2 = ''
    # (60 domain specific clauses)
    for d in range(10):
        atoms = [atom2idx[atom] for atom in [f'in({d},1)', f'digit(0,{d})', f'digit(1,{d})', f'digit(2,{d})', f'digit(3,{d})', f'digit(4,{d})']]
        cnf2 += f'-{atoms[0]} {atoms[1]} {atoms[2]} {atoms[3]} {atoms[4]} {atoms[5]} 0 \n'
        numClauses += 1
        for i in range(5):
            atoms = [atom2idx[atom] for atom in [f'in({d},0)', f'digit({i},{d})']]
            cnf2 += f'-{atoms[0]} -{atoms[1]} 0\n'
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

"""
CNF for b(x)+iSTE/sSTE
Write a clause below for all d in {0,...,9}
-in(d, true) v digit(0, d) v digit(1, d) v digit(2, d) v digit(3, d) v digit(4, d)
-in(d, false) v -digit(0, d)
-in(d, false) v -digit(1, d)
-in(d, false) v -digit(2, d)
-in(d, false) v -digit(3, d)
-in(d, false) v -digit(4, d)

Write clauses for UEC rule for img in {0,1,2,3,4} and i,j in {0,...,9} such that i<j
digit(img,0) v ... v digit(img,9)
-digit(img,i) v -digit(img,j)
"""

def write_cnf_UEC(path_cnf, path_atom2idx):
    # initialize all atoms and assign an index to each of them
    atom2idx = {}
    idx = 1 # the indices start from 1, which follows the dimacs format
    # define 5*10 atoms for digit/2
    for i in range(5):
        for d in range(10):
            atom2idx[f'digit({i},{d})'] = idx
            idx += 1
    # define 10*2 atoms for in/2
    for d in range(10):
        for b in range(2):
            atom2idx[f'in({d},{b})'] = idx
            idx += 1
    # construct the CNF
    numClauses = 0
    cnf2 = ''
    # (60 domain specific clauses)
    for d in range(10):
        atoms = [atom2idx[atom] for atom in [f'in({d},1)', f'digit(0,{d})', f'digit(1,{d})', f'digit(2,{d})', f'digit(3,{d})', f'digit(4,{d})']]
        cnf2 += f'-{atoms[0]} {atoms[1]} {atoms[2]} {atoms[3]} {atoms[4]} {atoms[5]} 0 \n'
        numClauses += 1
        for i in range(5):
            atoms = [atom2idx[atom] for atom in [f'in({d},0)', f'digit({i},{d})']]
            cnf2 += f'-{atoms[0]} -{atoms[1]} 0\n'
            numClauses += 1

    # (5 clauses for EC) add 5 rules of the form digit(img,0) v ... v digit(img,9)
    for img in ('0', '1', '2', '3', '4'):
        atoms = [str(atom2idx[f'digit({img},{n})']) for n in range(10)]
        atoms = ' '.join(atoms)
        cnf2 += f'{atoms} 0\n'
        numClauses += 1

    # (5*45=225 clauses for UC) add a rule for img in {0,1,2,3,4} and i,j in {0,...,9} such that i<j
    for img in ('0', '1', '2', '3', '4'):
        atomPairs = [[f'digit({img},{i})', f'digit({img},{j})'] for i in range(9) for j in range(i+1, 10)]
        atomPairs = [[str(atom2idx[i]), str(atom2idx[j])] for i,j in atomPairs]
        for i,j in atomPairs:
            cnf2 += f'-{i} -{j} 0\n'
            numClauses += 1

    cnf1 = f'p cnf {idx-1} {numClauses}\n'
    with open(path_cnf, 'w') as f:
        f.write(cnf1 + cnf2)
    json.dump(atom2idx, open(path_atom2idx,'w'))
    return atom2idx

def read_cnf_UEC(path_cnf, path_atom2idx):
    try:
        cnf = CNF(dimacs=path_cnf)
        atom2idx = json.load(open(path_atom2idx))
    except:
        atom2idx = write_cnf_UEC(path_cnf, path_atom2idx)
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
        @param b: a string in {B, Bi, Bs} denoting 3 combinations of b(x) and STE
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
        if b == 'B':
            self.C, self.atom2idx = read_cnf('./cnf/member5.cnf', './cnf/member5.atom2idx')
        else:
            self.C, self.atom2idx = read_cnf_UEC('./cnf/member5_UEC.cnf', './cnf/member5_UEC.atom2idx')
        self.numClauses, self.numAtoms = self.C.shape
        self.mapping = {'Bi': Bi, 'Bs': Bs}
        self.C = self.C.to(device)
        # self.apply(weight_init)

    def forward(self, x):
        """
        @param x: NN input of shape (batch_size, 5, 1, 28, 28)
        """
        # each data instance in a batch has 5 images; we group all images in a batch together
        x = x.view(-1, 1, 28, 28) # (batch_size*5, 1, 28, 28)
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x # (batch_size*5, 10)

    def do_gradient_update(self, data, debug, device, hyper, b):
        """
        @param data: a tuple (images, digit, label) where 
                     images is a tensor of shape (5, 1, 28, 28) denoting 5 digit images
                     digit is an integer in {0, ..., 9}
                     label is an integer in {0, 1} denoting False and True
        @param debug: a Boolean value denoting whether printing out more info for debugging
        @param hyper: a list of float numbers denoting hyper-parameters
        @param b: a string in {B, Bi, Bs} denoting 3 combinations of b(x) and STE
        """
        images, digit, label = data
        output = self.forward(images.to(device))
        output_bound = output[:]
        output_bound.retain_grad()
        output_cnf = output[:]
        output_cnf.retain_grad()
        losses = []
        if self.cnf: 
            C, v, g = self.vg_gen(output_cnf, digit, label, b, device)
            # C, v, g = self.vg_gen(output, digit, label, b, device)
            losses.append(reg_cnf(C, v, g) * hyper[0])
        if self.bound:
            losses.append(reg_bound(output_bound) * hyper[1])
            # losses.append(reg_bound(output) * hyper[1])
        loss = sum(losses)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()
        if debug:
            print(f'digit: {digit}')
            print(f'label: {label}')
            print('output')
            print(output[0])
            print('output_cnf.grad')
            print(output_cnf.grad[0])
            print('output_bound.grad')
            print(output_bound.grad[0])
        return loss
    
    # this version works with reg_cnf and requires batchSize = 1
    def vg_gen(self, output, digit, label, b, device):
        batch_size = label.shape[0]
        g = torch.zeros((batch_size, self.numAtoms), device=device)
        v = torch.zeros((batch_size, self.numAtoms), device=device)

        # 1. encode given information from (digit, label) into g
        for bidx in range(batch_size):
            idx = self.atom2idx[f'in({digit[bidx]},{label[bidx]})'] - 1
            g[bidx,idx] = 1
        
        if b == 'B':
            # 2. turn raw NN output into probabilities
            output = F.softmax(output, dim=-1).view(batch_size, 50) # (batch_size, 50)

            # 3. construct v from NN output and g
            # 3.1 first 50 atoms in v represent digit/2
            v[:,:50] = B(output)
            # 3.2 overwrite v with g
            v = g + v * (g==0).int()

            # 4. simplify C by filtering out the unrelated clauses in C
            # (only implemented for batch_size=1)
            C = self.C[self.C[:,idx]==-1]
            return C, v, g
        # In case b is "Bi" or "Bs"
        # 2. construct v using NN output and g
        # 2.1 turn raw NN output into {0,1}
        output = self.mapping[b](output).view(batch_size, 50) # (batch_size, 50)
        # 2.2 first 50 atoms in v represent digit/2
        v[:,:50] = output
        # 2.3 overwrite v with g
        v = g + v * (g==0).int()
        # 3. simplify C by filtering out the unrelated clauses in C
        # 3.1 the first 40 clauses are label related
        firstC = self.C[:60,:]
        secondC = self.C[60:,:]
        # 3.2 only simplify firstC by filtering out the unrelated clauses in it
        firstC = firstC[firstC[:,idx]==-1]
        C = torch.cat((firstC, secondC), 0)
        return C, v, g

    # this version works with reg_cnf_general and allows batchSize >= 1
    def vg_gen_general(self, output, digit, label, device):
        batch_size = label.shape[0]
        g = torch.zeros((batch_size, self.numAtoms), device=device)
        v = torch.zeros((batch_size, self.numAtoms), device=device)

        # 1. encode given information from label into g
        # where the last 20 atoms are in/2
        g[:,-20:] = F.one_hot(digit * 2 + label, num_classes=20)
        
        # 2. turn raw NN output into probabilities
        output = F.softmax(output, dim=-1).view(batch_size, 50) # (batch_size, 50)

        # 3. construct v from NN output and g
        # 3.1 first 50 atoms in v represent digit/2
        v[:,:50] = B(output)
        # 3.2 overwrite v with g
        v = g + v * (g==0).int()

        # 4. simplify C by filtering out the unrelated clauses in C
        mask = (torch.matmul(g, self.C.T) == -1).unsqueeze(-1) # (batch_size, m, 1); m=60
        C = self.C.unsqueeze(0) # (1, m, n)
        C = C * mask # (batch_size, m, n)
        return C, v, g

#############################
# Construct the training and testing data
#############################

class MNIST_Member(Dataset):

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
        i1, i2, i3, i4, i5, d, l = self.data[index]
        return torch.cat((self.dataset[i1][0], self.dataset[i2][0], self.dataset[i3][0], self.dataset[i4][0], self.dataset[i5][0]), 0).unsqueeze(1), d, l

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])
# used for training
trainDataset = MNIST_Member(torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform), './data/member5_train.txt')
# used to check training and testing accuracy
testLoader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, transform=transform), batch_size=1000, shuffle=True)
trainLoader = torch.utils.data.DataLoader(Subset(torchvision.datasets.MNIST('./data/', train=True, transform=transform), range(1000)), batch_size=1000)