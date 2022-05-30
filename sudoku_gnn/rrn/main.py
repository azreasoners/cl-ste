"""
Usage:
conda activate rrn-dgl
python main.py --output_dir out/ --do_train --batch_size 16 --batch_label 16 --num_train 10000 --gpu 0 --loss cross cont cnf bound
"""

from sudoku_data import sudoku_dataloader, sudoku_dataloader_semi
import argparse
from sudoku import SudokuRRN, SudokuCNN
import torch
from torch.optim import Adam
import os
import numpy as np

import sys
from collections import defaultdict
sys.path.append('../..')

from sudoku_gnn.ste import B, reg_cnf, reg_bound, reg_sudoku_cont
from sudoku_gnn.sudoku_cnf import read_cnf
from helper import test_RRN, testNN_trick

def vg_gen(idx, probs, inp):
    """Generate v and g for the output in first step

    Return:
        v: a tensor of shape (batch_size, 729), consisting of {0,1}
        g: a tensor of shape (batch_size, 729), consisting of {0,1}
        # probs: a tensor of shape (batch_size, 729) denoting the NN predictions related to v
    Args:
        idx: the index of the message passing step to apply STE loss
        probs: probabilities of shape (steps, batch_size, 9, 9, 9) where steps=32
        inp: NN input of shape (batch_size, 81)
    """
    batch_size = inp.shape[0]
    # 1. encode given information from inp into g
    g = torch.nn.functional.one_hot(inp.to(torch.int64), num_classes=10) # (batch_size, 81, 10)
    g = g[:,:,1:] # prediction on digit 0 is not considered in the CNF
    g = g.reshape(batch_size, 729).float()
    # 2. obtain the prediction only for step idx
    probs = probs[idx,...].reshape(batch_size, 729) # (batch_size, 729)
    # 3. construct v from NN output and g
    v = g + B(probs) * (g==0).int() # (batch_size, 729)
    return v, g

def main(args):
    if args.wandb:
        import wandb
        # We provide visualization through wandb. To use it, please register for an account "ste"
        # and replace the XXXX below with the your API key in https://wandb.ai/settings
        # os.environ['WANDB_API_KEY'] = 'XXXX'
        # wandb.init(project=f'DGL-{args.nn.upper()}', entity='ste')
        os.environ['WANDB_API_KEY'] = 'f448b1734f515fd253f7b6a0e7c9bce5e0508945'
        wandb.init(project=f'DGL-{args.nn.upper()}-03-2022', entity='arg')
    torch.manual_seed(args.seed)

    if args.gpu < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
        print('Using CPU')
    else:
        device = torch.device('cuda', args.gpu)
        print(f'Using GPU {args.gpu}')

    if args.nn == 'rrn':
        model = SudokuRRN(num_steps=args.steps, edge_drop=args.edge_drop)
    elif args.nn == 'cnn':
        model = SudokuCNN()
    else:
        print(f'Error: The specified nn {args.nn} is undefined. Please use either rrn or cnn.')
        sys.exit()

    if args.do_train:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        model.to(device)
        prefix = '_'.join(args.loss) + f'_{args.num_train // 1000}k'
        if args.wandb: wandb.watch(model, log_freq=100)
        C, _ = read_cnf('../../cnf/sudoku.cnf', '../../cnf/sudoku.atom2idx')
        C = C.to(device)
        print('The used STE losses: {}'.format(args.loss))
        if args.batch_label < args.batch_size:
            train_dataloader, train_dataloader_ulb = sudoku_dataloader_semi(
                args.batch_size,
                segment='train',
                num=args.num_train,
                batch_label=args.batch_label
                )
        else:
            train_dataloader = sudoku_dataloader(args.batch_size, segment='train', num=args.num_train)
            train_dataloader_ulb = [None] * len(train_dataloader)
        dev_dataloader = sudoku_dataloader(batch_size=32, segment='valid', num=1000)
        test_dataloader = sudoku_dataloader(batch_size=32, segment='test', num=1000)
        # define adjacency matrices for continuous regularizer reg_sudoku_cont
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

        opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_dev_acc = 0.0
        use_STE_loss = any([l != 'cross' for l in args.loss])
        # print(('{:<8}'+'{:<15}' * 4).format('epoch', 'total_acc', 'single_acc', 'total_count', 'single_count'))
        # row_format = '{:<8}' + '{:<15.2f}' * 2 + '{:<15}' * 2
        for epoch in range(args.epochs):
            model.train()
            losses = defaultdict(list)
            for i, (g, g_ulb) in enumerate(zip(train_dataloader, train_dataloader_ulb)):
                g = g.to(device)
                """
                g.ndata['q']: (batch_size * 81) where batch_size=64
                g.ndata['a']: (batch_size * 81)
                g.ndata['row']: (batch_size * 81)
                g.ndata['col']: (batch_size * 81)
                logits: (steps * batch_size * 81, 10) where steps = 32
                """
                loss = 0
                inp = g.ndata['q'].view(-1, 81) # (batch_size, 81)
                label = g.ndata['a'].view(-1, 81) # (batch_size, 81)
                if args.nn == 'rrn':
                    _, L_cross, logits = model(g)
                    prob = torch.nn.functional.softmax(logits, dim=-1)
                    prob = prob.view(-1,9,9,10)[...,1:] # (steps * batch_size, 9, 9, 9)
                    v, g = vg_gen(args.idx, prob.view(args.steps,-1,9,9,9), inp) # (batch_size, 729)
                else:
                    L_cross, logits = model(inp, label)
                    prob = torch.nn.functional.softmax(logits, dim=-1)
                    prob = prob.view(-1,9,9,9) # (batch_size, 9, 9, 9)
                    v, g = vg_gen(args.idx, prob.view(1,-1,9,9,9), inp) # (batch_size, 729)
                L_bound = reg_bound(logits) * args.alpha
                L_cont = reg_sudoku_cont(prob, A_row, A_col, A_box, A_same)
                L_sat, L_unsat, L_deduce = reg_cnf(C, v, g)
                L_cnf = L_sat + L_unsat + L_deduce
                for l in ('cross', 'cont', 'bound', 'deduce', 'sat', 'unsat', 'cnf'):
                    val = eval(f'L_{l}')
                    losses[l].append(val.item())
                    if l in args.loss:
                        loss += val
                
                if use_STE_loss and g_ulb:
                    g_ulb = g_ulb.to(device)
                    inp_ulb = g_ulb.ndata['q'].view(-1, 81) # (batch_size, 81)
                    _, _, logits_ulb = model(g_ulb)
                    L_bound_ulb = reg_bound(logits_ulb) * args.alpha
                    prob_ulb = torch.nn.functional.softmax(logits_ulb, dim=-1)
                    prob_ulb = prob_ulb.view(-1,9,9,10)[...,1:] # (steps * batch_size, 9, 9, 9)
                    L_cont_ulb = reg_sudoku_cont(prob_ulb, A_row, A_col, A_box, A_same) 
                    v_ulb, g_ulb = vg_gen(args.idx, prob_ulb.view(args.steps,-1,9,9,9), inp_ulb)
                    L_sat_ulb, L_unsat_ulb, L_deduce_ulb = reg_cnf(C, v_ulb, g_ulb)
                    L_cnf_ulb = L_sat_ulb + L_unsat_ulb + L_deduce_ulb
                    for l in ('cont', 'bound', 'deduce', 'sat', 'unsat', 'cnf'):
                        if l in args.loss:
                            loss += eval(f'L_{l}_ulb')

                opt.zero_grad()
                loss.backward()
                opt.step()
                if i % 100 == 0:
                    print(f"Epoch {epoch}, batch {i}, loss {loss.cpu().data}")                

            # compute loss average
            if args.wandb:
                for k in losses:
                    wandb.log({f'train_loss/{k}': sum(losses[k]) / len(losses[k])}, step=epoch)
                wandb.log({'logits_start': logits[0,-1]}, step=epoch)
                wandb.log({'logits_end': logits[-1,-1]}, step=epoch)

            # dev
            print("\n=========Dev step========")
            # update the number of message steps for testing
            model.rrn.num_steps = args.test_steps

            dev_acc, dev_loss = test_RRN(model, dev_dataloader, nn=args.nn)
            test_acc, _ = test_RRN(model, test_dataloader, nn=args.nn)
            train_acc, _ = test_RRN(model, train_dataloader, nn=args.nn, limit=100)
            test_acc_trick = testNN_trick(model, test_dataloader)
            # print(f'test acc with trick:\t{test_acc_trick}')
            # print(row_format.format(epoch+1, 100*correct/total, 100*singleCorrect/singleTotal,\
            #       f'{correct}/{total}', f'{singleCorrect}/{singleTotal}'))
            print(f"Dev loss {dev_loss}, accuracy {dev_acc}")
            if dev_acc >= best_dev_acc:
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_best.bin'))
                best_dev_acc = dev_acc
            print(f"Best dev accuracy {best_dev_acc}\n")
            # we save every CNN model to test with the inference trick
            if args.nn == 'cnn':
                torch.save(model.state_dict(), os.path.join(args.output_dir+'/cnn', f'{prefix}_{epoch}.pt'))

            if args.wandb:
                wandb.log({'train_acc': train_acc}, step=epoch)
                wandb.log({'val_acc': dev_acc}, step=epoch)
                wandb.log({'test_acc': test_acc}, step=epoch)
                wandb.log({'test_acc_trick': test_acc_trick}, step=epoch)
            
            # update the number of message steps for training
            model.rrn.num_steps = args.steps

        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_final.bin'))

    if args.do_eval:
        model_path = os.path.join(args.output_dir, 'model_best.bin')
        if not os.path.exists(model_path):
            raise FileNotFoundError("Saved model not Found!")

        model.load_state_dict(torch.load(model_path))
        model.to(device)
        
        test_dataloader = sudoku_dataloader(args.batch_size, segment='test')

        print("\n=========Test step========")
        model.eval()
        test_loss = []
        test_res = []
        for g in test_dataloader:
            g = g.to(device)
            target = g.ndata['a']
            target = target.view([-1, 81])

            with torch.no_grad():
                preds, loss = model(g, is_training=False)
                preds = preds
                preds = preds.view([-1, 81])

                for i in range(preds.size(0)):
                    test_res.append(int(torch.equal(preds[i, :], target[i, :])))

                test_loss.append(loss.cpu().detach().data)

        test_acc = sum(test_res) / len(test_res)
        print(f"Test loss {np.mean(test_loss)}, accuracy {test_acc}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recurrent Relational Network on sudoku task.')
    parser.add_argument("--output_dir", type=str, default=None, required=True,
                        help="The directory to save model")
    parser.add_argument("--do_train", default=False, action="store_true",
                        help="Train the model")
    parser.add_argument("--do_eval", default=False, action="store_true",
                        help="Evaluate the model on test data")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--edge_drop", type=float, default=0.4,
                        help="Dropout rate at edges.")
    parser.add_argument("--steps", type=int, default=32,
                        help="Number of message passing steps.")
    parser.add_argument('--test_steps', type=int, default=32,
                        help='the number of message passing steps during testing')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="weight decay (L2 penalty)")
    

    parser.add_argument('--loss', default=['cross'], nargs='+',
        help='specify regularizers in \{cross, deduce, sat, unsat, bound, uec\}')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='the weight for L_bound to control the size of raw NN output')
    parser.add_argument('--idx', type=int, default=0,
                        help='the index of the iteration to apply CNF loss')
    parser.add_argument('--num_train', type=int, default=-1,
                        help='the number of training data')
    parser.add_argument('--batch_label', type=int, default=16,
                        help='the number of labels in each batch')
    parser.add_argument('--nn', type=str, default='rrn',
                        help='the type of nn: rrn or cnn')


    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for reproductivity.')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='set the debug mode')
    parser.add_argument('--wandb', default=False, action='store_true',
                        help='save all logs on wandb')

    args = parser.parse_args()
    # we do not log onto wandb in debug mode
    if args.debug: args.wandb = False

    main(args)
