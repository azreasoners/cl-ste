import itertools
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sat import CNF
from ste import B, reg_bound, reg_cnf

torch.manual_seed(0) # manually set the seed to fix the train/test data

######################################
# Write the CNF Program if Not Exists
#   Definitions of the CNF for Sudoku where
#   atom a(R,C,N) represents "number N is assigned at row R column C"
######################################

"""
1. UEC on row index for each column C and each value N
1.1 EC constraint
a(1,C,N) v ... v a(9,C,N)
1.2 UC constraint for i,j in {0,...,9} such that i<j
-a(i,C,N) v -a(j,C,N)

2. UEC on column index for each row R and each value N
2.1 EC constraint
a(R,1,N) v ... v a(R,9,N)
2.2 UC constraint for i,j in {0,...,9} such that i<j
-a(R,i,N) v -a(R,j,N)

# THIS IS NOT NECESSARY DUE TO SOFTMAX
3. UEC on 9 values for each row R and each column C
3.1 EC constraint
a(R,C,1) v ... v a(R,C,9)
3.2 UC constraint for i,j in {0,...,9} such that i<j
-a(R,C,i) v -a(R,C,j)

4. UEC on 9 cells in the same 3*3 box for each value N
4.1 EC constraints
Disj_{i in {1,2,3} and j in {1,2,3}} a(i,j,N)
Disj_{i in {1,2,3} and j in {4,5,6}} a(i,j,N)
Disj_{i in {1,2,3} and j in {7,8,9}} a(i,j,N)
Disj_{i in {4,5,6} and j in {1,2,3}} a(i,j,N)
Disj_{i in {4,5,6} and j in {4,5,6}} a(i,j,N)
Disj_{i in {4,5,6} and j in {7,8,9}} a(i,j,N)
Disj_{i in {7,8,9} and j in {1,2,3}} a(i,j,N)
Disj_{i in {7,8,9} and j in {4,5,6}} a(i,j,N)
Disj_{i in {7,8,9} and j in {7,8,9}} a(i,j,N)
4.2 UC constraints for every 2 atoms a(i1,j1,N) and a(i2,j2,N) in a EC clause
-a(i1,j1,N) v -a(i2,j2,N)
"""

def write_cnf(path_cnf, path_atom2idx):
    # initialize all atoms and assign an index to each of them
    atom2idx = {}
    idx = 1 # the indices start from 1, which follows the dimacs format

    # define 9*9*9 atoms for a(R,C,N)
    for r in range(1,10):
        for c in range(1,10):
            for n in range(1,10):
                atom2idx[f'a({r},{c},{n})'] = idx
                idx += 1

    # construct the CNF
    numClauses = 0
    cnf2 = ''
    # 1. UEC on row index for each column C and each value N
    for c in range(1,10):
        for n in range(1,10):
            atoms = [str(atom2idx[f'a({r},{c},{n})']) for r in range(1,10)]
            # (EC) add a rule "a(1,C,N) v ... v a(9,C,N)"
            ec = ' '.join(atoms)
            cnf2 += f'{ec} 0\n'
            numClauses += 1
            # (UC) add "-a(i,C,N) v -a(j,C,N)" for all combinations of atoms
            for i,j in itertools.combinations(atoms, 2):
                cnf2 += f'-{i} -{j} 0\n'
                numClauses += 1

    # 2. UEC on column index for each row R and each value N
    for r in range(1,10):
        for n in range(1,10):
            atoms = [str(atom2idx[f'a({r},{c},{n})']) for c in range(1,10)]
            # (EC) add a rule "a(R,1,N) v ... v a(R,9,N)"
            ec = ' '.join(atoms)
            cnf2 += f'{ec} 0\n'
            numClauses += 1
            # (UC) add "-a(R,i,N) v -a(R,j,N)" for all combinations of atoms
            for i,j in itertools.combinations(atoms, 2):
                cnf2 += f'-{i} -{j} 0\n'
                numClauses += 1

    # # THIS IS NOT NECESSARY DUE TO SOFTMAX
    # # 3. UEC on 9 values for each row R and each column C
    # for r in range(1,10):
    #     for c in range(1,10):
    #         atoms = [str(atom2idx[f'a({r},{c},{n})']) for n in range(1,10)]
    #         # (EC) add a rule "a(R,C,1) v ... v a(R,C,9)"
    #         ec = ' '.join(atoms)
    #         cnf2 += f'{ec} 0\n'
    #         numClauses += 1
    #         # (UC) add "-a(R,C,i) v -a(R,C,j)" for all combinations of atoms
    #         for i,j in itertools.combinations(atoms, 2):
    #             cnf2 += f'-{i} -{j} 0\n'
    #             numClauses += 1

    # 4. UEC on 9 cells in the same 3*3 box for each value N
    positions = ((1,2,3), (4,5,6), (7,8,9))
    for n in range(1,10):
        for rs in positions:
            for cs in positions:
                atoms = [str(atom2idx[f'a({r},{c},{n})']) for r in rs for c in cs]
                # (EC) add a rule "a(R1,C1,N) v ... v a(R2,C2,N)" for atoms in the same box
                ec = ' '.join(atoms)
                cnf2 += f'{ec} 0\n'
                numClauses += 1
                # (UC) add "-a(R1,C1,N) v -a(R2,C2,N)" for all combinations of atoms
                for i,j in itertools.combinations(atoms, 2):
                    cnf2 += f'-{i} -{j} 0\n'
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
        @param bn: a boolean value denoting whether batch normalization is used

        Remark:
            The same CNN from [Park 2018] for solving Sudoku puzzles given as 9*9 matrices
            https://github.com/Kyubyong/sudoku
        """
        super(Net, self).__init__()
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
        self.cross, self.cnf, self.bound = [s in reg for s in ('cross', 'cnf', 'bound')]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.C, self.atom2idx = read_cnf('./cnf/sudoku.cnf', './cnf/sudoku.atom2idx')
        self.numClauses, self.numAtoms = self.C.shape
        self.apply(weight_init)
        if self.cnf: self.C = self.C.to(device)

    def forward(self, x):
        """
        @param x: NN input of shape (batch_size, 1, 9, 9)
        """
        x = self.nn(x) # (batch_size, 9, 9, 9)
        x = x.permute(0,2,3,1)
        x = x.view(-1,81,9)
        return x # (batch_size, 81, 9)

    def do_gradient_update(self, data, debug, device, hyper, b):
        """
        @param data: a tuple (config, label) where 
                     config is a tensor of shape (batch_size, 1, 9, 9)
                     label is a tensor of shape (batch_size, 81)
        @param debug: a Boolean value denoting whether printing out more info for debugging
        @param hyper: a list of float numbers denoting hyper-parameters
        @param b: a string in {B, Bi, Bs} denoting 3 combinations of b(x) and STE
        """
        config, label = data
        config, label = config.to(device), label.to(device)
        output = self.forward(config.to(device)) # (batch_size, 81, 9)
        losses = []
        if self.cross:
            losses.append(F.cross_entropy(output.reshape(-1, output.shape[-1]), label.view(-1)))
        if self.cnf: 
            v, g = self.vg_gen(output, config, label, device)
            losses.append(reg_cnf(self.C, v, g) * hyper[0])
        if self.bound: 
            losses.append(reg_bound(output) * hyper[1])
        loss = sum(losses)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def vg_gen(self, output, config, label, device):
        batch_size = label.shape[0]
        # 1. encode given information from config into g
        g = F.one_hot(config.to(torch.int64), num_classes=10)[..., 1:].reshape(batch_size,729).float() # {0,1} of shape (batchSize, 729)
        # 2. turn raw NN output into probabilities
        output = F.softmax(output, dim=-1).view(batch_size, 729) # (batch_size,729)
        # 3. construct v from NN output and g
        v = g + B(output) * (g==0).int()
        return v, g

#############################
# Construct the training and testing data
#############################

def load_pickle(file_to_load):
    with open(file_to_load, 'rb') as fp:
        labels = pickle.load(fp)
    return labels

class Sudoku_dataset(Dataset):
    def __init__(self, input_path, label_path, data_limit=None):
        input_dict = load_pickle(input_path) # 130k keys, each mapped to a (9,9) np array
        label_dict = load_pickle(label_path) # 130k keys, each mapped to a (9,9) np array
        keys_to_keep = list(set(input_dict).intersection(set(label_dict)))[:data_limit]
        self.data = []
        for k in keys_to_keep:
            x = torch.from_numpy(input_dict[k]).unsqueeze(0).float() # (1,9,9) integers from 0~9
            y = torch.from_numpy(label_dict[k]).float() - 1 # (9,9) integers from 0~8
            y = y.reshape(81).long()
            self.data.append((x, y))
        print(f'The whole dataset contains {len(self.data)} data instances')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


input_file = './data/easy_130k_given.p'
label_file = './data/easy_130k_solved.p'

# Construct the dataset of 70k data instances
dataset = Sudoku_dataset(input_file, label_file, data_limit=70000)

# Split the dataset into train/val/test
train_size = int(0.8 * len(dataset))
test_size = int(0.2 * len(dataset))
trainDataset, testDataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Construct the dataloaders (used to check training and testing accuracy) for the datasets
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=32, shuffle=True)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size=32, shuffle=False)