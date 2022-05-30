"""
Usage:
conda activate ste
python main.py --epochs 20 --gpu 0 --loss cross cnf bound cont --num_examples 30000
"""

import argparse
from collections import defaultdict
import networkx as nx
import torch
from torch import nn
import pandas as pd
import pickle
import random

import os
import wandb
# We provide visualization through wandb. To use it, please register for an account
# and replace the XXXX below with the your API key in https://wandb.ai/settings
os.environ['WANDB_API_KEY'] = 'XXXX'

import sys 
sys.path.append('../..')
from sudoku_gnn.ste import B, reg_bound, reg_hint, reg_cnf, reg_sudoku_cont
from sudoku_gnn.sudoku_cnf import read_cnf

########################################################################
# Graph Neural Network for Sudoku
########################################################################

class GNNTransformer(nn.Module):
    def __init__(self):
        super(GNNTransformer, self).__init__()
        G = nx.sudoku_graph() # there are 1620 edges, i.e., 1620 ones in nx.adjacency_matrix(G).todense()
        # If a BoolTensor is provided, positions with True is not allowed to attend while False values will be unchanged
        self.register_buffer("A", torch.BoolTensor(1 - nx.adjacency_matrix(G).todense()))
        hidden_dim = 128
        num_heads = 4
        num_layers = 8

        # node embedding to higher dimension
        self.embedding = nn.Linear(10, hidden_dim)

        # aggregation layers
        self.transformers = nn.ModuleList([nn.MultiheadAttention(hidden_dim, num_heads) for _ in range(num_layers)])
        self.pre_norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.post_norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # mlp
        self.mlp = nn.Sequential(*[
                                   nn.Linear(hidden_dim, hidden_dim*2),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim * 2, 9)
        ])

    def forward(self, batch):
        # batch is a matrix of size b x 81 x 10
        x = self.embedding(batch) # (batch_size, 81, 128)
        residual = x # (batch_size, 81, 128)
        # x1 will store the updated x after 1 message passing
        x1 = None
        for transformer, pre_norm_layer, post_norm_layer in zip(self.transformers, self.pre_norm_layers, self.post_norm_layers):
            x = pre_norm_layer(x) # (batch_size, 81, 128)
            # input requirements: sequence length (81) x batch x hidden_dim
            x = x.transpose(0,1) # (81, batch_size, 128)
            x, _ = transformer(x, x, x, attn_mask=self.A) # (81, batch_size, 128)
            # input requirements: batch x sequence length (81) x hidden_dim
            x = x.transpose(0,1) # (batch_size, 81, 128)
            x = residual + nn.functional.relu(post_norm_layer(x))
            residual = x
            if x1 is None:
                x1 = x
        logits = self.mlp(x).reshape(-1, 9)
        logits1 = self.mlp(x1).reshape(-1, 9)
        return logits, logits1


########################################################################
# Data Preparation
########################################################################

# For 9M dataset

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, dataset):
        # dataset is a list of tuples (x, y)
        self.dataset = dataset

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset)

  def __getitem__(self, index):
        # Load data and get label
        data = self.dataset[index]
        X = data[0]
        y = data[1]
        return X, y

def encode(list_x):
    x = torch.LongTensor(list_x)
    x_encoded = torch.zeros((81, 10), dtype=torch.float)
    x_encoded[torch.arange(81), x.reshape(-1)] = 1
    return x_encoded

def dataGen(args):
    # Load dataset
    print(f'loading {args.num_examples} number of data from 9M data instances')
    # Since the 9M dataset is too big, we place it with a 100k subset
    # df = next(pd.read_csv('./sudoku.csv', chunksize=(args.num_examples)))
    df = next(pd.read_csv('./sudoku_100k_from_9m.csv', chunksize=(args.num_examples)))

    puzzle, solution = df[["puzzle", "solution"]].values.T # np.array of strs of len 81

    print('creating data instances...') # each data is ((81,10), (81))
    graphs = [(encode([int(d) for d in p]), 
            torch.LongTensor([int(d)-1 for d in s])) for p,s in zip(puzzle, solution)]

    print('shuffling data instances...')
    random.shuffle(graphs)

    print('splitting dataset to train and test: ', end='')
    train_9m = graphs[:int(args.num_examples * args.train_split)]
    val_9m = graphs[int(args.num_examples * args.train_split):]
    print(len(train_9m), len(val_9m))

    # Prepare train dataloader
    if args.batch_label == args.batch_size:
        # supervised learning
        training_set = Dataset(train_9m)
        trainloader_lb = torch.utils.data.DataLoader(
            training_set,
            batch_size=args.batch_size,
            shuffle=True)
        trainloader_ulb = [None] * len(trainloader_lb)
    else:
        print('splitting training data into supervised and unsupervised:')
        split_idx = int(len(train_9m) * (args.batch_label / args.batch_size))
        train_lb = train_9m[:split_idx]
        train_ulb = train_9m[split_idx:]
        training_set_lb = Dataset(train_lb)
        training_set_ulb = Dataset(train_ulb)
        trainloader_lb = torch.utils.data.DataLoader(
            training_set_lb,
            batch_size=args.batch_label,
            shuffle=True)
        trainloader_ulb = torch.utils.data.DataLoader(
            training_set_ulb,
            batch_size=args.batch_size - args.batch_label,
            shuffle=True)

    # Prepare test dataloader
    test_set = Dataset(val_9m)
    testloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False)
    return trainloader_lb, trainloader_ulb, testloader

# For 70k dataset

def load_pickle(file_to_load):
    with open(file_to_load, 'rb') as fp:
        labels = pickle.load(fp)
    return labels

class Dataset_70k(Dataset):
    def __init__(self, input_path, label_path, data_limit=None):
        input_dict = load_pickle(input_path) # 130k keys, each mapped to a (9,9) np array
        label_dict = load_pickle(label_path) # 130k keys, each mapped to a (9,9) np array
        keys_to_keep = list(set(input_dict).intersection(set(label_dict)))[:data_limit]
        self.data = []
        for k in keys_to_keep:
            x = encode(input_dict[k])
            y = torch.from_numpy(label_dict[k]).reshape(81).long() - 1 # (9,9) integers from 0~8
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Dataset_palm(Dataset):
    """
    Note:
        The Palm train/test/val dataset contain 180k, 18k, 18k puzzles with
        a uniform distribution of givens between 17 and 34 (which are 18 kinds)
    """
    def __init__(self, dataset_path, given=None):
        with open(dataset_path) as f:
            df = pd.read_csv(dataset_path, header=None)
            puzzle, solution = df.values.T
            self.data = [(encode([int(d) for d in p]), 
                torch.LongTensor([int(d)-1 for d in s])) for p,s in zip(puzzle, solution)]
            # reader = csv.reader(f, delimiter=',')
            # self.data = [(sudoku_str2tensor(q).view(1,9,9).float(), sudoku_str2tensor(a)-1) for q, a in reader]
        if given:
            self.data = [(inp, label) for (inp, label) in self.data if ((inp != 0).sum().item() in given)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def dataGen_70k(args):
    input_file = '../../data/easy_130k_given.p'
    label_file = '../../data/easy_130k_solved.p'
    # Construct the dataset of 70k data instances
    dataset_70k = Dataset_70k(input_file, label_file, data_limit=70000)
    # Split the dataset into train/val/test
    train_size = int(0.8 * len(dataset_70k))
    test_size = int(0.2 * len(dataset_70k))
    train_dataset_70k, test_dataset_70k = torch.utils.data.random_split(dataset_70k, [train_size, test_size])

    # Construct the dataloaders (used to check training and testing accuracy) for the datasets
    trainloader_lb = torch.utils.data.DataLoader(
        train_dataset_70k,
        batch_size=args.batch_size,
        shuffle=True)
    trainloader_ulb = [None] * len(trainloader_lb)
    testloader_70k = torch.utils.data.DataLoader(
        test_dataset_70k,
        batch_size=args.batch_size,
        shuffle=False)

    test_dataset_palm = Dataset_palm('../../data/palm_sudoku/test.csv')
    testloader_palm = torch.utils.data.DataLoader(
        test_dataset_palm,
        batch_size=args.batch_size,
        shuffle=False)
    return trainloader_lb, trainloader_ulb, testloader_70k, testloader_palm

########################################################################
# Helper functions
########################################################################

def vg_gen(inp, probs):
    """Generate v and g given NN input inp and output probs

    Return:
        v: a tensor of shape (batch_size, 729), consisting of {0,1}
        g: a tensor of shape (batch_size, 729), consisting of {0,1}
    Args:
        inp: NN input of shape (batch_size, 81, 10)
        probs: probabilities of shape (batch_size, 9, 9, 9)
    """
    g = inp[...,1:].reshape(-1,729) # (batch_size, 729)
    v = g + B(probs).view(-1,729) * (g==0).int() # (batch_size, 729)
    return v, g

def test(net, dataloader, device, max_data=None, mask=False):
    """
    Args:
        net: NN model
        dataloader: (DataLoader) the (data, label) pairs to be tested
        max_data: (int) the number of maximum number of data to be tested
        mask: (bool) whether to only consider empty cells when evaluating board acc
              note that we only consider empty cells when evaluating cell acc
    """
    net.eval()
    total = 0
    correct = 0
    singleCorrect = 0
    singleTotal = 0
    for en, (data, label) in enumerate(dataloader):
        if max_data and en + 1 == max_data:
            break
        data, label = data.to(device), label.to(device)
        output, _ = net(data) # R^(batch_size * 81, 9)
        output = output.view(-1, 81, 9) # R^(batch_size, 81, 9)

        if label.shape == output.shape[:-1]:
            pred = output.argmax(dim=-1) # {0,...,8}^(batch_size, 81)
        elif label.shape == output.shape:
            pred = (output >= 0).int()
        else:
            print(f'Error: none considered case for output with shape {output.shape} v.s. label with shape {label.shape}')
            import sys
            sys.exit()
        
        # create the mask to filter in empty cells only
        m = data.argmax(dim=-1) == 0 # {T,F}^(batch_size, 81)

        if len(label.shape) == 1:
            correct += (label.int() == pred.int()).all().int().item()
            total += 1
        else:
            # {True, False}^(batch_size, 81)
            correctionMatrix = (label.int() == pred.int()).view(label.shape[0], -1)
            if mask:
                correctionMatrix = m.int() * correctionMatrix.int() + (1 - m.int())
            correct += correctionMatrix.all(1).sum().item()
            total += label.shape[0]
            singleCorrect += correctionMatrix.sum().item() - (1 - m.int()).sum().item()
            singleTotal += m.int().sum().item()
    board_accuracy = correct * 100/total if total else 0
    cell_accuracy = singleCorrect * 100/singleTotal if singleTotal else 0
    return board_accuracy, cell_accuracy

########################################################################
# Main Function
########################################################################

def main(args):
    if args.wandb:
        semi = '-Semi' if args.batch_size > args.batch_label else ''
        park = '-70k' if args.park else ''
        wandb.init(project=f'Kaggle-GNN-Sudoku{semi}{park}', entity='ste')
    else:
        wandb.init(mode='disabled')

    # Seed everything for reproductivity
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Set up device: CPU or GPU
    if args.gpu < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
        print('Using CPU')
    else:
        device = torch.device('cuda', index=args.gpu)
        print(f'Using GPU {args.gpu}')

    # Load data
    if args.park:
        trainloader_lb, trainloader_ulb, testloader, testloader2 = dataGen_70k(args)
    else:
        trainloader_lb, trainloader_ulb, testloader = dataGen(args)

    # Initialize NN model, define optimizer and baseline loss function
    net = GNNTransformer()
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # define CNF C
    C, _ = read_cnf('../../cnf/sudoku.cnf', '../../cnf/sudoku.atom2idx')
    C = C.to(device)

    # define adjacency matrices for continuous regularizer loss_sudoku_cont
    A_same = torch.eye(81, device=device)
    A_row = torch.zeros([81, 81], dtype=torch.float32, device=device)
    A_col = torch.zeros([81, 81], dtype=torch.float32, device=device)
    A_box = torch.zeros([81, 81], dtype=torch.float32, device=device)
    for i in range(81):
        for j in range(81):
            ix, iy = i // 9, i % 9
            jx, jy = j // 9, j % 9
            ic = 3 * (ix // 3) + iy // 3
            jc = 3 * (jx // 3) + jy // 3
            if i == j:
                continue
            if ix == jx:
                A_row[i, j] = 1
            if iy == jy:
                A_col[i, j] = 1
            if ic == jc:
                A_box[i, j] = 1


    ########################################################################
    # Start training
    ########################################################################

    print(f'The loss functions to be used are {args.loss}')
    if args.wandb:
        wandb.watch(net, log_freq=100)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        # we use a dictionary losses to record all loss values in every batch
        # where each kind of loss is a list of scalars
        losses = defaultdict(list)
        running_loss = 0.
        for i, (data, data_ulb) in enumerate(zip(trainloader_lb, trainloader_ulb)):
            loss = 0
            # get the inputs; data is a list [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            if data_ulb is not None:
                inputs_ulb = data_ulb[0].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs, outputs1 = net(inputs) # (batch_label * 81, 9)
            if data_ulb is not None:
                outputs_ulb, outputs1_ulb = net(inputs_ulb) # ((batch_size-batch_label) * 81, 9)

            # clone the outputs so that we can compute gradients separately
            outputs_cross = outputs.clone()
            outputs_ste = outputs.clone()
            outputs1_ste = outputs1.clone()
            outputs_cross.retain_grad()
            outputs_ste.retain_grad()
            outputs1_ste.retain_grad()

            # cross-entropy loss for labeled data
            L_cross = criterion(outputs_cross, labels.view(-1))

            #########################
            # STE losses are applied to both labeled and unlabeled data
            # all losses with postfix "_1" means they are applied to the NN prediction
            # after 1 message passing
            #########################

            if data_ulb is not None:
                outputs_all = torch.cat((outputs_ste, outputs_ulb), dim=0) # (batch_size*81, 9)
                outputs1_all = torch.cat((outputs1_ste, outputs1_ulb), dim=0) # (batch_size*81, 9)
                inputs_all = torch.cat((inputs, inputs_ulb), dim=0) # (batch_size, 81, 10)
            else:
                outputs_all = outputs_ste # (batch_size * 81, 9)
                outputs1_all = outputs1_ste # (batch_size * 81, 9)
                inputs_all = inputs # (batch_size, 81, 10)

            # obtain the probabilistic outputs
            y_pred = torch.nn.Softmax(dim=-1)(outputs_all).view(args.batch_size, 9, 9, 9)
            y_pred_1 = torch.nn.Softmax(dim=-1)(outputs1_all).view(args.batch_size, 9, 9, 9)
            v, g = vg_gen(inp=inputs_all, probs=y_pred) # (batch_size, 729)
            v_1, _ = vg_gen(inp=inputs_all, probs=y_pred_1) # (batch_size, 729)

            L_hint = reg_hint(y_pred.view(-1,729), g)
            L_cont = reg_sudoku_cont(y_pred, A_row, A_col, A_box, A_same)
            L_bound = reg_bound(outputs_all) * args.alpha
            L_sat, L_unsat, L_deduce = reg_cnf(C, v, g)
            L_cnf = L_sat + L_unsat + L_deduce

            for l in ('cross', 'hint', 'cont', 'bound', 'sat', 'unsat', 'deduce', 'cnf'):
                val = eval(f'L_{l}')
                losses[l].append(val.item())
                if l in args.loss: loss += val

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                print('[%d, %5d] train loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

        ########################################################################
        # Testing after each epoch
        ########################################################################

        # compute train accuracy
        train_acc_lb, train_acc_lb_cell = test(net, trainloader_lb, device, 1000, mask=args.mask)
        if args.wandb:
            wandb.log({'accuracy/train_lb': train_acc_lb}, step=epoch)
            wandb.log({'accuracy/train_lb_cell': train_acc_lb_cell}, step=epoch)
        print(f'[Train] Board acc on labeled data: {train_acc_lb:.3f}')
        print(f'[Train] Cell acc on labeled data: {train_acc_lb_cell:.3f}')

        if args.batch_label < args.batch_size:
            train_acc_ulb, train_acc_ulb_cell = test(net, trainloader_ulb, device, 1000, mask=args.mask)
            if args.wandb:
                wandb.log({'accuracy/train_ulb': train_acc_ulb}, step=epoch)
                wandb.log({'accuracy/train_ulb_cell': train_acc_ulb_cell}, step=epoch)
            print(f'[Train] Board acc on un-labeled data: {train_acc_ulb:.3f}')
            print(f'[Train] Cell acc on un-labeled data: {train_acc_ulb_cell:.3f}')
        
        # compute test accuracy
        test_acc, test_acc_cell = test(net, testloader, device, mask=args.mask)
        if args.wandb:
            wandb.log({'accuracy/test': test_acc}, step=epoch)
            wandb.log({'accuracy/test_cell': test_acc_cell}, step=epoch)
        print(f'[Test] Board acc: {test_acc:.3f}')
        print(f'[Test] Cell acc: {test_acc_cell:.3f}')

        if args.park:
            test_acc, test_acc_cell = test(net, testloader2, device, mask=args.mask)
            if args.wandb:
                wandb.log({'accuracy/test_palm': test_acc}, step=epoch)
                wandb.log({'accuracy/test_cell_palm': test_acc_cell}, step=epoch)
            print(f'[Test Palm] Board acc: {test_acc:.3f}')
            print(f'[Test Palm] Cell acc: {test_acc_cell:.3f}')

        # compute loss average
        if args.wandb:
            for k in losses:
                wandb.log({f'train_loss/{k}': sum(losses[k]) / len(losses[k])}, step=epoch)
            wandb.log({'logits': outputs_all[0,0]}, step=epoch)
            wandb.log({'logits1': outputs1_all[0,0]}, step=epoch)
            wandb.log({'grad/g_cross': outputs_cross.grad[0,0]}, step=epoch)
            if outputs_ste.grad is not None:
                wandb.log({'grad/g_ste': outputs_ste.grad[0,0]}, step=epoch)
            if outputs1_ste.grad is not None:
                wandb.log({'grad/g_ste1': outputs1_ste.grad[0,0]}, step=epoch)

    print('Finished Training')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Total number of epoch.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--batch_label', type=int, default=16,
                        help='the number of training data with labels in each batch')
    parser.add_argument('--num_examples', type=int, default=30000, help='The number of examples in the whole dataset')
    parser.add_argument('--train_split', type=float, default=0.8, help='the portion of data for training')
    parser.add_argument('--loss', default=['cross'], nargs='+',
        help='specify regularizers in \{cross, hint, cont, deduce, sat, unsat, cnf, bound\}')
    parser.add_argument('--park', default=False, action='store_true', help='do the Park 70k experiments')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='the weight for L_bound to control the size of raw NN output')
    
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproductivity.')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu index; -1 means using CPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay (L2 penalty)')
    parser.add_argument('--mask', default=False, action='store_true', help='if True, only consider empty cells when testing')
    parser.add_argument('--debug', default=False, action='store_true', help='debug mode')
    parser.add_argument('--wandb', default=False, action='store_true', help='save all logs on wandb')
    args = parser.parse_args()

    if args.num_examples == -1:
        print('Please make sure that the 100k subset dataset has been replaced with the full 9M dataset.')
        args.num_examples = 9000000
    # we do not log onto wandb in debug mode
    if args.debug: args.wandb = False
    main(args)