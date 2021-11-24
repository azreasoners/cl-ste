import itertools
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset

from sat import CNF
from ste import B, reg_bound, reg_cnf
from network import FC


######################################
# Write the CNF Program if Not Exists
######################################

"""
Each node is represented by a pair (i,j) for i,j in {0,...,3}
The edges connected to node (i,j) are represented by pairs (whenever valid)
  ((i,j),(i+1,j)), ((i,j),(i-1,j)), ((i,j),(i,j+1)), ((i,j),(i,j-1))
An atom "terminal(i,j)" represents that node (i,j) is a terminal node in the SP
An atom "sp((i,j), (i+di,j+dj))" represents that an edge is in the SP

Note that sp((1,2), (0,2)) and sp((0,2), (1,2)) are treated as the same atom in the CNF

# The CNF simply represents: "terminal nodes are connected to 1 edge"
Write a clause below for i,j in {0,...,3} (for terminal nodes)
-terminal(i,j) v V_{di,dj}sp((i,j), (i+di,j+dj))
Write a clause below for i,j in {0,...,3}, and for different (di,dj) and (di2,dj2)
-terminal(i,j) v -sp((i,j), (i+di,j+dj)) v -sp((i,j), (i+di2,j+dj2))
"""

def write_cnf(path_cnf, path_atom2idx):
    # initialize all atoms and assign an index to each of them
    atom2idx = {}
    idx = 1 # the indices start from 1, which follows the dimacs format
    connected_edges = {} # map an atom "terminal(i,j)" to a list of atoms "sp((i,j),(i+di,j+dj))"

    # define 4*4 atoms for terminal(i,j)
    for i in range(4):
        for j in range(4):
            atom2idx[f'terminal({i},{j})'] = idx
            connected_edges[idx] = [] # records the edges connected to node (i,j)
            idx += 1

    # define 24 atoms for sp((i,j), (i+di,j+dj))
    for i in range(4):
        for j in range(4):
            for di,dj in ((0,1), (0,-1)):
                newi, newj = i+di, j+dj
                new_atom1 = f'sp(({i},{j}),({newi},{newj}))'
                new_atom2 = f'sp(({newi},{newj}),({i},{j}))'
                if 0 <= newi < 4 and 0 <= newj < 4 and new_atom1 not in atom2idx.keys():
                    atom2idx[new_atom1] = idx
                    atom2idx[new_atom2] = idx
                    connected_edges[atom2idx[f'terminal({i},{j})']].append(idx)
                    connected_edges[atom2idx[f'terminal({newi},{newj})']].append(idx)
                    idx += 1
    for j in range(4):
        for i in range(4):
            for di,dj in ((1,0), (-1,0)):
                newi, newj = i+di, j+dj
                new_atom1 = f'sp(({i},{j}),({newi},{newj}))'
                new_atom2 = f'sp(({newi},{newj}),({i},{j}))'
                if 0 <= newi < 4 and 0 <= newj < 4 and new_atom1 not in atom2idx.keys():
                    atom2idx[new_atom1] = idx
                    atom2idx[new_atom2] = idx
                    connected_edges[atom2idx[f'terminal({i},{j})']].append(idx)
                    connected_edges[atom2idx[f'terminal({newi},{newj})']].append(idx)
                    idx += 1

    # construct the CNF
    numClauses = 0
    cnf2 = ''

    # add 16 rules of the form "-terminal(i,j) v V_{di,dj}sp((i,j), (i+di,j+dj))"
    for i in range(4):
        for j in range(4):
            terminal = atom2idx[f'terminal({i},{j})']
            sp_atoms = connected_edges[terminal]
            sp_atoms = ' '.join([str(atom) for atom in sp_atoms])
            cnf2 += f'-{terminal} {sp_atoms} 0\n'
            numClauses += 1

    # add 52 rules of the form "-terminal(i,j) v -sp((i,j), (i+di,j+dj)) v -sp((i,j), (i+di2,j+dj2))"
    for i in range(4):
        for j in range(4):
            terminal = atom2idx[f'terminal({i},{j})']
            sp_atoms = connected_edges[terminal]
            sp_pairs = list(itertools.combinations(sp_atoms, 2))
            for sp1, sp2 in sp_pairs:
                cnf2 += f'-{terminal} -{sp1} -{sp2} 0\n'
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
        self.nn = FC(40, 50, 50, 50, 50, 50, 24)
        self.cross, self.cnf, self.bound = [s in reg for s in ('cross', 'cnf', 'bound')]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.C, self.atom2idx = read_cnf('./cnf/shortestPath.cnf', './cnf/shortestPath.atom2idx')
        self.numClauses, self.numAtoms = self.C.shape
        self.apply(weight_init)
        self.C = self.C.to(device)
        # self.scale = 1

    def forward(self, x):
        """
        @param x: NN input of shape (batch_size, 40)
        """
        return self.nn(x) # (batch_size, 24)

    def do_gradient_update(self, data, debug, device, hyper, b):
        """
        @param data: a tuple (config, label) where config is of shape (40) denoting a
                     grid configuration and 
                     label is of shape (24)
        @param debug: a Boolean value denoting whether printing out more info for debugging
        @param hyper: a list of float numbers denoting hyper-parameters
        @param b: a string in {B, Bi, Bs} denoting 3 combinations of b(x) and STE
        """
        config, label = data
        label = label.to(device)
        output = self.forward(config.to(device))
        output.retain_grad()
        losses = []
        if self.cross:
            output_c = output[:]
            output_c.retain_grad()
            losses.append(nn.functional.binary_cross_entropy_with_logits(output_c, label))
        if self.cnf: 
            v, g = self.vg_gen(config, output, label, device)
            v.retain_grad()
            losses.append(reg_cnf(self.C, v, g) * hyper[0])
        if self.bound: 
            output_b = output[:]
            output_b.retain_grad()
            losses.append(reg_bound(output_b) * hyper[1])
        loss = sum(losses)
        loss.backward(retain_graph=True)
        self.optimizer.step()

        if debug:
            print(f'min:{output.min().item()}\tmax:{output.max().item()}')

        self.optimizer.zero_grad()
        return loss

    def vg_gen(self, config, output, label, device):
        batch_size = label.shape[0]
        g = torch.zeros((batch_size, self.numAtoms), device=device)
        v = torch.zeros((batch_size, self.numAtoms), device=device)

        # 1. encode given information from label into g
        # where the first 16 atoms in CNF are terminal(i,j) for 16 nodes
        g[:,:16] = config[:,24:]

        # 2. Turn raw output into probabilities
        output = torch.sigmoid(output) # (batch_size, 24)

        # 3. encode predicted information from NN output to v
        # where the last 24 atoms in CNF are sp/2 for 24 edges
        v[:,16:] = B(output)

        # 4. merge g and v to construct the final prediction v
        v = g + v * (g==0).int()
        return v, g

#############################
# Construct the training and testing data
#############################

class GridData(Dataset):

    def __init__(self, examples):
        self.data = list() # data stored as integers for the ease of reading
        self.dataset = list()
        with open(examples) as f:
            for line in f:
                line = line.strip().split(',')
                remove = [int(i) for i in line[0].split('-')]
                endNodes = [int(i) for i in line[1].split('-')]
                label = [int(i) for i in line[2].split('-')]
                self.data.append((remove, endNodes, label))
                inp = torch.cat((self.oneHot(remove, 24, inv=True), self.oneHot(endNodes, 16)), 0) # input to NN is of shape (40)
                self.dataset.append((inp, self.oneHot(label, 24)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.dataset[index]

    @staticmethod
    def oneHot(numbers, n, inv=False):
        one_hot = torch.zeros(n)
        one_hot[numbers] = 1
        if inv:
            one_hot = (one_hot + 1) % 2
        return one_hot

dataset = GridData('data/shortestPath.data')
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
torch.manual_seed(0) # manually set the seed for reproducibility
trainDataset, testDataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# used to check training and testing accuracy
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=32, shuffle=True)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size=1000, shuffle=False)