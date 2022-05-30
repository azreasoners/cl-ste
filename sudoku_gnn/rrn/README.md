## Introduction
The codes in this folder are the Recurrent Relational Network (RRN) codes reimplemented in [DGL repository](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rrn).
* Paper link: https://arxiv.org/abs/1711.08028
* Author's code repo: https://github.com/rasmusbergpalm/recurrent-relational-networks

## How to run
Below are the commands for supervised and semi-supervised experiments.
```
python main.py --output_dir out/ --do_train --batch_size 16 --batch_label 16 --num_train 10000 --gpu 0 --loss cross cont cnf bound

python main.py --output_dir out/ --do_train --batch_size 16 --batch_label 8 --num_train 20000 --gpu 0 --loss cross cont cnf bound
```

## Visualization (Optional)
We provide visualization through wandb. To use it, please register for an account and replace the XXXX in code `os.environ['WANDB_API_KEY'] = 'XXXX'` in line 78 with the your API key in https://wandb.ai/settings. If you specify `--wandb` in your command, you will be able to check the logged loss values and accuracy at https://wandb.ai/home.