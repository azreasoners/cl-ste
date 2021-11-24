import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from sat import CNF
from ste import B, Bi, Bs, reg_bound, reg_cnf


######################################
# Write the CNF Program if Not Exists
######################################

"""
CNF for b_p(x)+iSTE
Write a clause below for all d0, d1, d2 in {0,...,9} and for all r
-apply(d0,d1,d2,r) v Disjuction_{o0,o1 | r=(d0 o0 d1) o1 d2}(operators(o0,o1))
"""

def write_cnf(path_cnf, path_atom2idx):
    # initialize all atoms and assign an index to each of them
    atom2idx = {}
    idx = 1 # the indices start from 1, which follows the dimacs format
    disjunction = {} # a mapping from "apply(d0,d1,d2,r)" to a list of all possible "operators(o0,o1)"
    posR = {} # a mapping from (d0, d1, d2) to a set of all possible results r

    # define 9 atoms for operators/2
    for o0 in range(3):
        for o1 in range(3):
            atom2idx[f'operators({o0},{o1})'] = idx
            idx += 1
    # define 10597 atoms for apply/4
    for d0 in range(11):
        for d1 in range(11):
            for d2 in range(11):
                for o0 in range(3):
                    for o1 in range(3):
                        if o0 == 0:
                            r1 = d0 + d1
                        elif o0 == 1:
                            r1 = d0 - d1
                        elif o0 == 2:
                            r1 = d0 * d1
                        if o1 == 0:
                            r = r1 + d2
                        elif o1 == 1:
                            r = r1 - d2
                        elif o1 == 2:
                            r = r1 * d2
                        if (d0,d1,d2) in posR:
                            posR[(d0,d1,d2)].add(r)
                        else:
                            posR[(d0,d1,d2)] = set([r])
                        atom = f'apply({d0},{d1},{d2},{r})'
                        if atom not in atom2idx:
                            atom2idx[atom] = idx
                            idx += 1
                        if atom in disjunction:
                            disjunction[atom].append(f'operators({o0},{o1})')
                        else:
                            disjunction[atom] = [f'operators({o0},{o1})']

    # construct the CNF
    numClauses = 0
    cnf2 = ''
    # add the rules "-apply(d0,d1,d2,r) v Disjuction_{o0,o1 | r=(d0 o0 d1) o1 d2}(operators(o0,o1))"
    for d0 in range(11):
        for d1 in range(11):
            for d2 in range(11):
                for r in posR[(d0,d1,d2)]:
                    applyAtom = f'apply({d0},{d1},{d2},{r})'
                    disj = ' '.join([str(atom2idx[atom]) for atom in disjunction[applyAtom]])
                    cnf2 += f'{disj} -{atom2idx[applyAtom]} 0\n'
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
Write a clause below for all d0, d1, d2 in {0,...,9} and for all r
-apply(d0,d1,d2,r) v Disjuction_{o0,o1 | r=(d0 o0 d1) o1 d2}(operators(o0,o1))

Write clauses for UEC rule for img in {i1,i2} and i,j in {0,1,2} such that i<j
operator(img,0) v operator(img,1) v operator(img,2)
-operator(img,i) v -operator(img,j)
"""

def write_cnf_UEC(path_cnf, path_atom2idx):
    # initialize all atoms and assign an index to each of them
    atom2idx = {}
    idx = 1 # the indices start from 1, which follows the dimacs format
    disjunction = {} # a mapping from "apply(d0,d1,d2,r)" to a list of all possible "operators(o0,o1)"
    posR = {} # a mapping from (d0, d1, d2) to a set of all possible results r

    # define 2*3=6 atoms for operator/2
    for img in ('i1', 'i2'):
        for o in range(3):
            atom2idx[f'operator({img},{o})'] = idx
            idx += 1
    # define 9 atoms for operators/2
    for o0 in range(3):
        for o1 in range(3):
            atom2idx[f'operators({o0},{o1})'] = idx
            idx += 1
    # define 10597 atoms for apply/4
    for d0 in range(11):
        for d1 in range(11):
            for d2 in range(11):
                for o0 in range(3):
                    for o1 in range(3):
                        if o0 == 0:
                            r1 = d0 + d1
                        elif o0 == 1:
                            r1 = d0 - d1
                        elif o0 == 2:
                            r1 = d0 * d1
                        if o1 == 0:
                            r = r1 + d2
                        elif o1 == 1:
                            r = r1 - d2
                        elif o1 == 2:
                            r = r1 * d2
                        if (d0,d1,d2) in posR:
                            posR[(d0,d1,d2)].add(r)
                        else:
                            posR[(d0,d1,d2)] = set([r])
                        atom = f'apply({d0},{d1},{d2},{r})'
                        if atom not in atom2idx:
                            atom2idx[atom] = idx
                            idx += 1
                        if atom in disjunction:
                            disjunction[atom].append(f'operators({o0},{o1})')
                        else:
                            disjunction[atom] = [f'operators({o0},{o1})']

    # construct the CNF
    numClauses = 0
    cnf2 = ''

    # add the UEC rule
    for img in ('i1', 'i2'):
        # add the rule "operator(img,0) v operator(img,1) v operator(img,2)"
        atoms = [f'operator({img},{i})' for i in range(3)]
        atoms = ' '.join([str(atom2idx[atom]) for atom in atoms])
        cnf2 += f'{atoms} 0\n'
        numClauses += 1
        # add the clauses "-operator(img,i) v -operator(img,j)" for i,j in {0,1,2} such that i<j
        for i in range(2):
            for j in range(i+1,3):
                atom1 = atom2idx[f'operator({img},{i})']
                atom2 = atom2idx[f'operator({img},{j})']
                cnf2 += f'-{atom1} -{atom2} 0\n'
                numClauses += 1

    # add the rules "-apply(d0,d1,d2,r) v Disjuction_{o0,o1 | r=(d0 o0 d1) o1 d2}(operators(o0,o1))"
    for d0 in range(11):
        for d1 in range(11):
            for d2 in range(11):
                for r in posR[(d0,d1,d2)]:
                    applyAtom = f'apply({d0},{d1},{d2},{r})'
                    disj = ' '.join([str(atom2idx[atom]) for atom in disjunction[applyAtom]])
                    cnf2 += f'{disj} -{atom2idx[applyAtom]} 0\n'
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
        torch.manual_seed(786)
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    if isinstance(m, nn.Conv2d):
        torch.manual_seed(786)
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
                nn.Linear(16 * 5 * 5, 120),
                nn.BatchNorm1d(120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.BatchNorm1d(84),
                nn.ReLU(),
                nn.Linear(84, 3)
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
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 3)
            )
        self.cnf, self.bound = [s in reg for s in ('cnf', 'bound')]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if b == 'B':
            self.C, self.atom2idx = read_cnf('./cnf/apply2x2.cnf', './cnf/apply2x2.atom2idx')
        else:
            self.C, self.atom2idx = read_cnf_UEC('./cnf/apply2x2_UEC.cnf', './cnf/apply2x2_UEC.atom2idx')
        self.numClauses, self.numAtoms = self.C.shape
        self.mapping = {'Bi': Bi, 'Bs': Bs}
        self.C = self.C.to(device)
        # self.apply(weight_init)

    def forward(self, x):
        """
        @param x: NN input of shape (batch_size, 4, 1, 32, 32)
        """
        # each data instance in a batch has 4 images; we group all images in a batch together
        x = x.view(-1, 1, 32, 32) # (batch_size*4, 1, 32, 32)
        x = self.encoder(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x # (batch_size*4, 3)

    def do_gradient_update(self, data, debug, device, hyper, b):
        """
        @param data: a tuple (images, d1, d2, d3, r1, r2, c1, c2) where 
                     images is of shape (4, 1, 32, 32) denoting 4 images of operators in {+, -, x}
                     d1, d2, d3 are integers in {0, ..., 9}
                     r1, r2, c1, c2 are integers denoting the results on rows/cols
        @param debug: a Boolean value denoting whether printing out more info for debugging
        @param hyper: a list of float numbers denoting hyper-parameters
        @param b: a string in {B, Bi, Bs} denoting 3 combinations of b(x) and STE
        """
        images, d1, d2, d3, r1, r2, c1, c2 = data
        output = self.forward(images.to(device))
        losses = []
        if self.cnf: 
            output = output.view(-1, 4, 3)
            losses.extend((reg_cnf(*self.vg_gen(output[:,[0,1]], d1, d2, d3, r1, b, device)),
                           reg_cnf(*self.vg_gen(output[:,[2,3]], d1, d2, d3, r2, b, device)),
                           reg_cnf(*self.vg_gen(output[:,[0,2]], d1, d2, d3, c1, b, device)),
                           reg_cnf(*self.vg_gen(output[:,[1,3]], d1, d2, d3, c2, b, device))))
            if hyper[0] != 1:
                losses = [i * hyper[0] for i in losses]
        if self.bound: 
            losses.append(reg_bound(output) * hyper[1])
        loss = sum(losses)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()
        if debug:
            print('output[0][0]')
            print(output[0][0])
        return loss

    # this version works with reg_cnf and requires batchSize = 1
    def vg_gen(self, output, d0, d1, d2, r, b, device):
        batch_size = r.shape[0]
        g = torch.zeros((batch_size, self.numAtoms), device=device)
        v = torch.zeros((batch_size, self.numAtoms), device=device)

        # 1. encode given information from (d0, d1, d2, r) into g
        for bidx in range(batch_size):
            idx = self.atom2idx[f'apply({d0[bidx]},{d1[bidx]},{d2[bidx]},{r[bidx]})'] - 1
            g[bidx,idx] = 1

        if b == 'B':
            # 2. turn raw NN output into probabilities
            output = F.softmax(output, dim=-1).view(batch_size, 2, 3) # (batch_size, 2, 3)

            # 3. construct v from NN output and g
            # 3.1 the first 9 atoms in v represent operators/2
            v[:,:9] = B(torch.matmul(output[:,0].unsqueeze(2), output[:,1].unsqueeze(1)).view(batch_size, 9))
            # 3.2 overwrite v with g
            v = g + v * (g==0).int()

            # 4. simplify C by filtering out the unrelated clauses in C 
            # (only implemented for batch_size=1)
            C = self.C[self.C[:,idx]==-1]
            return C, v, g

        # In case b is "Bi" or "Bs"
        # 2. construct v from NN output and g
        # 2.1 turn raw NN output into {0,1}
        output = self.mapping[b](output).view(batch_size, 2, 3) # (batch_size, 2, 3)
        # 2.2 first 6 atoms in v represent operator/2
        v[:,:6] = output.view(batch_size, 6) # (batch_size, 6)
        # 2.3 following 9 atoms in v represent operators/2
        v[:,6:15] = torch.matmul(output[:,0].unsqueeze(2), output[:,1].unsqueeze(1)).view(batch_size, 9)
        # 2.4 overwrite v with g
        v = g + v * (g==0).int()

        # 3. simplify C by filtering out the unrelated clauses in C (only implemented for batch_size=1)
        #    where the first 8 clauses are for UEC
        firstC = self.C[:8,:]
        secondC = self.C[8:,:]
        C = torch.cat((firstC, secondC[secondC[:,idx]==-1]), 0)
        return C, v, g

#############################
# Construct the training and testing data
#############################

class Hasy_Apply(Dataset):

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
        d1, d2, d3, o1, o2, o3, o4, r1, r2, c1, c2 = self.data[index] # self.dataset[o1][0] is of shape (1,32,32)
        return torch.cat((self.dataset[o1][0], self.dataset[o2][0], self.dataset[o3][0], self.dataset[o4][0]), 0).unsqueeze(1), d1, d2, d3, r1, r2, c1, c2

transform = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5, ))])
hasy_train_data = torchvision.datasets.ImageFolder(root = './data/hasy/apply2x2_train', transform=transform)
hasy_test_data = torchvision.datasets.ImageFolder(root = './data/hasy/apply2x2_test', transform=transform)
# used for training
trainDataset = Hasy_Apply(hasy_train_data, './data/apply2x2_train.txt')
# used to check training and testing accuracy
testLoader = torch.utils.data.DataLoader(hasy_test_data, batch_size=1000, shuffle=True)
trainLoader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root = './data/hasy/apply2x2_train', transform=transform), batch_size=1000, shuffle=True)