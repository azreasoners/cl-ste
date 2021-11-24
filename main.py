import argparse
import importlib
import os
import time
import torch
import numpy as np

from network import train, testNN

def main(args):
    start_time = time.time()
    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')
    print(f'device is {device.type}')

    # import the module for the specified domain name
    domain = importlib.import_module(f'problems.{args.domain}')
    # start to run the same experiment for args.trials times and print accuracy for each trial
    for i in range(args.trials):
        model = domain.Net(reg=args.reg, lr=args.lr, bn=args.bn, device=device, b=args.b)
        if args.numData != 0 and args.numData < len(domain.trainDataset):
            np.random.seed(args.seed) # fix the random seed to fix the training data
            domain.trainDataset = torch.utils.data.Subset(domain.trainDataset, np.random.choice(len(domain.trainDataset), args.numData, replace=False))
        domain.trainDataset = torch.utils.data.DataLoader(domain.trainDataset, batch_size=args.batchSize, shuffle=True)
        print('accuracy\ttrain\ttest')
        train(model=model, trainDataset=domain.trainDataset, epochs=args.epochs, \
              save=args.save, folder=args.folder, load=args.load, \
              checkAcc=args.checkAcc, testLoader=domain.testLoader, trainLoader=domain.trainLoader, \
              device=device, debug=args.debug, plot=args.plot, bar=args.bar, hyper=args.hyper, b=args.b)
        # check testing accuracy
        accuracy, singleAccuracy = testNN(model=model, testLoader=domain.testLoader, device=device)
        # check training accuracy
        accuracyTrain, singleAccuracyTrain = testNN(model=model, testLoader=domain.trainLoader, device=device)
        if 'sudoku' in args.domain:
            print(f'cell    \t{singleAccuracyTrain:0.2f}\t{singleAccuracy:0.2f}')
            print(f'board   \t{accuracyTrain:0.2f}\t{accuracy:0.2f}')
        else:
            print(f'        \t{accuracyTrain:0.2f}\t{accuracy:0.2f}')
    print('--- total time from beginning: %s seconds ---' % int(time.time() - start_time) )

    # play a sound once complete
    if args.sound:
        try:
            os.system('say "Mission Complete."')
        except:
            print('\a')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='mnistAdd', help='domain name')
    parser.add_argument('--numData', type=int, default=0, help='number of data for training; 0 means all training data')
    parser.add_argument('--seed', type=int, default=1, help='the random seed used to sample training data')
    parser.add_argument('--trials', type=int, default=1, help='number of trials')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs in each trial')
    parser.add_argument('--batchSize', type=int, default=1, help='batch size for training')
    parser.add_argument('--reg', default=['cnf', 'bound'], nargs='+', help='<Required> specify regularizers in \{bound, cnf, cross\}')
    parser.add_argument('--lr', type=float, default=0.0005, help='the learning rate of NN')
    parser.add_argument('--hyper', default=[1, 0.1], nargs='+', type=float, help='Hyper parameters: Weights of [L_cnf, L_bound]')
    parser.add_argument('--b', type=str, default='B', choices=['B', 'Bi', 'Bs'], help='set up the combination of b(x) and STE')
    parser.add_argument('--bn', default=False, action='store_true', help='use batch normalization in the model')
    parser.add_argument('--checkAcc', default=False, action='store_true', help='check test accuracy for every epoch')
    parser.add_argument('--debug', default=False, action='store_true', help='indicate the debug mode, where more info would be printed out')
    parser.add_argument('--save', type=str, default='NA', help='if not NA, save the NN model trained at every epoch i into [save]_epoch_[i].pt')
    parser.add_argument('--folder', type=str, default='./trained', help='the path to store all trained models')
    parser.add_argument('--load', type=str, default='NA', help='if not NA, load the trained NN model named [folder]/[load].pt before training')
    parser.add_argument('--gpu', default=False, action='store_true', help='try to use GPU if cuda is available')
    parser.add_argument('--sound', default=False, action='store_true', help='play a sound when the process is done')
    parser.add_argument('--plot', type=int, default=0, help='if not 0, compute test accuracy for every [plot] iterations')
    parser.add_argument('--bar', default=False, action='store_true', help='show the progress bar during training')
    parser.add_argument('--comment', type=str, default='', help='placeholder for additional info for an experiment')
    args = parser.parse_args()
    main(args)
