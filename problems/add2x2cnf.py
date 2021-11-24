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
Write a clause below for l in {0,...,18} and for (i1,i2) in {(o0,o1), (o2,o3), (o0,o2), (o1,o3)}
-sum(i1,i2,l) v Disjunction_{d1,d2 | d1+d2=l} aux(i1,d1,i2,d2)
where the aux atom represents the conjunction of "i1 being d1" and "i2 being d2"
"""

def write_cnf(path_cnf, path_atom2idx):
    # initialize all atoms and assign an index to each of them
    atom2idx = {}
    idx = 1 # the indices start from 1, which follows the dimacs format
    # define 400 aux atoms
    for i,j in ((0,1), (2,3), (0,2), (1,3)):
        for v1 in range(10):
            for v2 in range(10):
                atom2idx[f'aux(o{i},{v1},o{j},{v2})'] = idx
                idx += 1
    # define 4*19 atoms for labels
    for i,j in ((0,1), (2,3), (0,2), (1,3)):
        for v in range(19):
            atom2idx[f'sum(o{i},o{j},{v})'] = idx
            idx += 1

    # construct the CNF
    numClauses = 0
    cnf2 = ''
    for i,j in ((0,1), (2,3), (0,2), (1,3)):
        # add 19 rules of the form -sum(oi,oj,l) v Disjunction_{d1,d2 | d1+d2=l} aux(oi,d1,oj,d2)
        for v in range(19):
            atom = atom2idx[f'sum(o{i},o{j},{v})']
            atoms = [str(atom2idx[f'aux(o{i},{v1},o{j},{v2})']) for v1 in range(10) for v2 in range(10) if v1+v2==v]
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


"""
CNF for b(x)+iSTE/sSTE
Write a clause below for l in {0,...,18} and for (i1,i2) in {(o0,o1), (o2,o3), (o0,o2), (o1,o3)}
-sum(i1,i2,l) v Disjunction_{d1,d2 | d1+d2=l} aux(i1,d1,i2,d2)

Write clauses for UEC rule for img in {o0,o1,o2,o3} and i,j in {0,...,9} such that i<j
digit(img,0) v ... v digit(img,9)
-digit(img,i) v -digit(img,j)
"""

def write_cnf_UEC(path_cnf, path_atom2idx):
    # initialize all atoms and assign an index to each of them
    atom2idx = {}
    idx = 1 # the indices start from 1, which follows the dimacs format

    # define 10*4 atoms for digit(img,n)
    for img in ('o0', 'o1', 'o2', 'o3'):
        for n in range(10):
            atom2idx[f'digit({img},{n})'] = idx
            idx += 1

    # define 400 aux atoms
    for i,j in ((0,1), (2,3), (0,2), (1,3)):
        for v1 in range(10):
            for v2 in range(10):
                atom2idx[f'aux(o{i},{v1},o{j},{v2})'] = idx
                idx += 1
    # define 4*19 atoms for labels
    for i,j in ((0,1), (2,3), (0,2), (1,3)):
        for v in range(19):
            atom2idx[f'sum(o{i},o{j},{v})'] = idx
            idx += 1
    # construct the CNF
    numClauses = 0
    cnf2 = ''
    for i,j in ((0,1), (2,3), (0,2), (1,3)):
        # add 19 rules of the form -sum(oi,oj,l) v Disjunction_{d1,d2 | d1+d2=l} aux(oi,d1,oj,d2)
        for v in range(19):
            atom = atom2idx[f'sum(o{i},o{j},{v})']
            atoms = [str(atom2idx[f'aux(o{i},{v1},o{j},{v2})']) for v1 in range(10) for v2 in range(10) if v1+v2==v]
            atoms = ' '.join(atoms)
            cnf2 += f'{atoms} -{atom} 0\n'
            numClauses += 1
    # add 4 rules of the form digit(img,0) v ... v digit(img,9)
    for img in ('o0', 'o1', 'o2', 'o3'):
        atoms = [str(atom2idx[f'digit({img},{n})']) for n in range(10)]
        atoms = ' '.join(atoms)
        cnf2 += f'{atoms} 0\n'
        numClauses += 1
    # add a rule for img in {o0,o1,o2,o3} and i,j in {0,...,9} such that i<j
    for img in ('o0', 'o1', 'o2', 'o3'):
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
            self.C, self.atom2idx = read_cnf('./cnf/add2x2.cnf', './cnf/add2x2.atom2idx')
        else:
            self.C, self.atom2idx = read_cnf_UEC('./cnf/add2x2_UEC.cnf', './cnf/add2x2_UEC.atom2idx')
        self.numClauses, self.numAtoms = self.C.shape
        self.mapping = {'Bi': Bi, 'Bs': Bs}
        self.C = self.C.to(device)
        # self.apply(weight_init)

    def forward(self, x):
        """
        @param x: NN input of shape (batch_size, 4, 1, 28, 28)
        """
        # each data instance in a batch has 4 images; we group all images in a batch together
        x = x.view(-1, 1, 28, 28) # (batch_size*4, 1, 28, 28)
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x # (batch_size*4, 10)

    def do_gradient_update(self, data, debug, device, hyper, b='B'):
        """
        @param data: a tuple (images, r1, r2, c1, c2) where 
                     images is of shape (4, 1, 28, 28) denoting 4 digit images
                     and r1, ..., c2 are integers denoting 4 sums
        @param debug: a Boolean value denoting whether printing out more info for debugging
        @param hyper: a list of float numbers denoting hyper-parameters
        @param b: a string in {B, Bi, Bs} denoting 3 combinations of b(x) and STE
        """
        images, r1, r2, c1, c2 = data
        output = self.forward(images.to(device))
        losses = []
        if self.cnf:
            label = torch.cat(
                (r1.unsqueeze(1), r2.unsqueeze(1),c1.unsqueeze(1),c2.unsqueeze(1)),
                dim=1
            ) # (batchSize, 4)
            C, v, g = self.vg_gen(output, label, b, device)
            losses.append(reg_cnf(C, v, g) * hyper[0])
        if self.bound: 
            losses.append(reg_bound(output) * hyper[1])
        loss = sum(losses)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()
        if debug:
            print('output[0]')
            print(output[0])
        return loss

    def vg_gen(self, output, label, b, device):
        batch_size = label.shape[0]
        g = torch.zeros((batch_size, self.numAtoms), device=device)
        v = torch.zeros((batch_size, self.numAtoms), device=device)

        # 1. encode given information from label into g
        # where the last 76 atoms are sum/3
        g[:,-76:] = F.one_hot(label, num_classes=19).view(batch_size, 76) # (batch_size, 76)

        if b == 'B':
            # 2. turn raw NN output into probabilities
            output = F.softmax(output, dim=-1).view(batch_size, 4, 10) # (batch_size, 4, 10)
            # 3. construct v from NN output and g
            # 3.1 the first 400 atoms in v represent aux/4
            vn = [torch.matmul(
                    output[:,i].unsqueeze(2),
                    output[:,j].unsqueeze(1)
                    ).view(batch_size, 100)
                  for i,j in ((0,1), (2,3), (0,2), (1,3))]
            v[:,:400] = B(torch.cat(vn, -1)) # (batch_size, 400)
            # 3.2 overwrite v with g
            v = g + v * (g==0).int()
            # 4. simplify C by filtering out the unrelated clauses in C 
            mask = torch.matmul(g, self.C.T) == -1 # (batch_size, m); m=76
            C = self.C.unsqueeze(0).repeat(batch_size,1,1) # (batch_size, m, n)
            C = C[mask] # (batch_size * m', n) where m' is the number of related clauses
            C = C.view(batch_size, 4, 476) # (batch_size, m', n) where m'=4
            return C, v, g

        # In case b is "Bi" or "Bs"
        # 2. turn raw NN output into {0,1}
        output = self.mapping[b](output).view(batch_size, 4, 10) # (batch_size, 4, 10)
        # 2.1 first 40 atoms in v represent digit/2
        v[:,:40] = output.view(batch_size, -1) # (batch_size, 40)
        # 2.2 following 400 atoms in v represent aux/4
        vn = [torch.matmul(
                output[:,i].unsqueeze(2),
                output[:,j].unsqueeze(1)
                ).view(batch_size, 100)
              for i,j in ((0,1), (2,3), (0,2), (1,3))]
        v[:,40:440] = torch.cat(vn, -1)
        # 2.3 overwrite v with g
        v = g + v * (g==0).int()

        # # indeed, we can directly return the complete C at this step since
        # # simplifying C does not save much time
        # return self.C, v, g

        # 3. simplify C by filtering out the unrelated clauses in C
        # 3.1 the first 76 clauses are label related
        firstC = self.C[:76,:]
        secondC = self.C[76:,:]
        # 3.2 only simplify firstC by filtering out the unrelated clauses in it
        mask = torch.matmul(g, firstC.T) == -1 # (batch_size, 76)
        firstC = firstC.unsqueeze(0).repeat(batch_size,1,1) # (batch_size, 76, n)
        firstC = firstC[mask].view(batch_size, 4, 516)
        secondC = secondC.unsqueeze(0).repeat(batch_size,1,1)
        C = torch.cat((firstC, secondC), 1)
        return C, v, g


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
        i1, i2, i3, i4, r1, r2, c1, c2 = self.data[index]
        return torch.cat((self.dataset[i1][0], self.dataset[i2][0], self.dataset[i3][0], self.dataset[i4][0]), 0).unsqueeze(1), r1, r2, c1, c2

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])
# used for training
trainDataset = MNIST_Addition(torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform), './data/add2x2_train.txt')
# used to check training and testing accuracy
testLoader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, transform=transform), batch_size=1000)
trainLoader = torch.utils.data.DataLoader(Subset(torchvision.datasets.MNIST('./data/', train=True, transform=transform), range(1000)), batch_size=1000)