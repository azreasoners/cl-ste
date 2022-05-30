## Introduction
This folder contains the codes for the supervised and semi-supervised experiments on the [Kaggle's GNN for Sudoku](https://www.kaggle.com/matteoturla/can-graph-neural-network-solve-sudoku).

## How to run
One can train the baseline GNN for 50 epochs with the following command where "--gpu 0" specifies the index of the GPU for this experiment. Use "--gpu -1" if you want to train on CPU.
```
python main.py --gpu 0 --epochs 50 --loss cross
```
One can also train the GNN with cross-entropy loss + STE losses with the following command.
```
python main.py --gpu 1 --epochs 50 --loss cross hint cont cnf bound
```
For semi-supervised experiments, one can use the following command to specify the size of the dataset and the number of data&labels in each batch.
```
python main.py --gpu 2 --epochs 50 --loss cross hint cont cnf bound --num_examples 60000 --batch_size 16 --batch_label 8
```

We list available loss functions below. 
- cross: baseline cross-entropy loss
- hint: the given digits should be predicted
- cont: the sum of 9 probabilities in every row/col/box in the prediction must be 1
- deduce: the L_deduce portion of L_cnf
- sat: the L_sat portion of L_cnf
- unsat: the L_unsat portion of L_cnf
- cnf: L_cnf
- bound: the size of raw NN prediction should be bounded
The above losses are applied to the final prediction (after 8 message passing steps). 

## Visualization (Optional)
We provide visualization through wandb. To use it, please register for an account and replace the XXXX in code `os.environ['WANDB_API_KEY'] = 'XXXX'` in line 20 with the your API key in https://wandb.ai/settings. If you specify `--wandb` in your command, you will be able to check the logged loss values and accuracy at https://wandb.ai/home.