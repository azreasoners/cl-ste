# STE for Learning
Injecting Logical Constraints into Neural Networks via Straight-Through-Estimators 

## Introduction
This repository provides the codes, instructions, and logs for all the experiments reported in the paper "Injecting Logical Constraints into Neural Networks via Straight-Through-Estimators".

## Installation
0. Install Anaconda according to its [installation page](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
1. Create a new environment using the following commands in terminal.
```bash
conda create -n ste python=3.7
conda activate ste
```
2. Install wandb for visualization, the DGL package for GNN experiments, and Pandas
```
pip install wandb dgl-cu102 dglgo -f https://data.dgl.ai/wheels/repo.html
conda install pandas
```
3. Install Pytorch according to its [Get-Started page](https://pytorch.org/get-started/locally/). Below is an example command we used on Linux with cuda 10.2.
```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
or for CPU only
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## File/Folder Description
```
.
├── main.py                 # The main script to train neural networks with semantic constraints using STE
├── ste.py                  # STE related classes and functions
├── network.py              # Neural network related classes and functions
├── sat.py                  # CNF related classes and functions
├── inference_trick.py      # The script to run inference trick for Sudoku problems
├── testBenchmark.sh        # The bash file to test all simple benchmark problems
├── problems                # The folder that includes multiple python scripts, one for each problem
├── sudoku_gnn              # The folder that includes the codes for Sudoku-GNN experiments
│   ├── rrn                 # The folder for RRN experiments
│   ├── gnn                 # The folder for typical GNN experiments
├── data                    # The folder that includes all the data
│   ├── hasy                # The operator dataset for apply2x2 problem from NeuroLog
│   ├── palm_sudoku         # The sudoku dataset from RRN (Palm)
│   ├── SATNet_sudoku       # The sudoku dataset from SATNet
│   ├── easy_130k_*.p       # The sudoku dataset from NeurASP (Park 70k)
│   ├── shortestPath.data   # The shortest path problem dataset from Semantic Loss
│   ├── *_train.txt         # The datasets either from NeuroLog or generated by us
├── cnf                     # The folder that includes CNF files for all problems
├── trained                 # The folder that includes all trained models
├── logs                    # The folder that includes some logs of experiments
└── README.md
```

## How to Run
You can test all simple benchmark problems by executing the following command.
```
bash testBenchmark.sh
```

You can also try the following commands to test individual problems. More descriptions about each option are available in `main.py`. The Sudoku GNN experiments are implemented in a separate folder "sudoku_gnn" to allow for the flexibility of both supervised and semi-supervised learning with different datasets.

1. mnistAdd, mnistAdd with batch size 16, mnistAdd2, mnistAdd3
```
python main.py --domain mnistAddcnf --epochs 1 --lr 0.001
python main.py --domain mnistAddcnf --epochs 1 --lr 0.001 --batchSize 16
python main.py --domain mnistAdd2cnf --epochs 1 --lr 0.001 --hyper 1 0.01
python main.py --domain mnistAdd3cnf --epochs 1 --lr 0.001 --hyper 1 0.001
```
Note that the 2 numbers after "--hyper" are the weights for L_cnf and L_bound. We fine-tune the weights in mnistAdd example to achieve the best accuracy within a single epoch. (We only need to adjust the weight of L_bound to make the absolute value of raw NN output not too big/small. Similar high accuracy could be achieved with a big range of weights of L_bound.) 

2. add2x2, apply2x2, member3, member5
```
python main.py --domain add2x2cnf --numData 3000
python main.py --domain apply2x2cnf --numData 3000
python main.py --domain member3cnf --numData 3000
python main.py --domain member5cnf --numData 3000
```

3. Sudoku CNN: [Park's CNN](https://github.com/Kyubyong/sudoku) training with STE and testing with [inference trick](https://github.com/Kyubyong/sudoku) on Park (sudokucnf) or Palm dataset (sudokuPalmcnf)
```
python main.py --domain sudokucnf --reg cnf bound --epochs 30 --gpu --bn --checkAcc --bar --folder ./trained --save sudoku
python inference_trick.py --gpu --bn --folder ./trained --domain sudokucnf
python inference_trick.py --gpu --bn --folder ./trained --domain sudokuPalmcnf
```
For the GNN experiments of Sudoku, please refer to the README.md file in "sudoku_gnn" folder.

4. Sudoku GNN
- [RRN from DGL](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rrn)

More instructions are available at `sudoku_gnn/rrn/README.md`.
```
cd sudoku_gnn/rrn
python main.py --output_dir out/ --do_train --batch_size 16 --batch_label 16 --num_train 10000 --gpu 0 --loss cross cont cnf bound
```

- [GNN from Kaggle](https://www.kaggle.com/matteoturla/can-graph-neural-network-solve-sudoku)

More instructions are available at `sudoku_gnn/gnn/README.md`.
```
cd sudoku_gnn/gnn
python main.py --epochs 20 --gpu 0 --loss cross cnf bound cont --num_examples 10000
```

5. shortestPath
```
python main.py --domain shortestPathcnf --reg cross cnf bound --epochs 500 --lr 0.002 --checkAcc --gpu --batchSize 32 --hyper 0.2 1
```
