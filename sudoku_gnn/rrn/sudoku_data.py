import csv
import os
import urllib.request
import zipfile
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataset import Dataset
import torch
import dgl
from copy import copy


def _basic_sudoku_graph():
    grids = [[0, 1, 2, 9, 10, 11, 18, 19, 20],
             [3, 4, 5, 12, 13, 14, 21, 22, 23],
             [6, 7, 8, 15, 16, 17, 24, 25, 26],
             [27, 28, 29, 36, 37, 38, 45, 46, 47],
             [30, 31, 32, 39, 40, 41, 48, 49, 50],
             [33, 34, 35, 42, 43, 44, 51, 52, 53],
             [54, 55, 56, 63, 64, 65, 72, 73, 74],
             [57, 58, 59, 66, 67, 68, 75, 76, 77],
             [60, 61, 62, 69, 70, 71, 78, 79, 80]]
    edges = set()
    for i in range(81):
        row, col = i // 9, i % 9
        # same row and col
        row_src = row * 9
        col_src = col
        for _ in range(9):
            edges.add((row_src, i))
            edges.add((col_src, i))
            row_src += 1
            col_src += 9
        # same grid
        grid_row, grid_col = row // 3, col // 3
        for n in grids[grid_row*3 + grid_col]:
            if n != i:
                edges.add((n, i))
    edges = list(edges)
    g = dgl.graph(edges)
    return g


class ListDataset(Dataset):
    def __init__(self, *lists_of_data):
        assert all(len(lists_of_data[0]) == len(d) for d in lists_of_data)
        self.lists_of_data = lists_of_data

    def __getitem__(self, index):
        return tuple(d[index] for d in self.lists_of_data)

    def __len__(self):
        return len(self.lists_of_data[0])


def _get_sudoku_dataset(segment='train'):
    assert segment in ['train', 'valid', 'test']
    url = "https://data.dgl.ai/dataset/sudoku-hard.zip"
    zip_fname = "/tmp/sudoku-hard.zip"
    dest_dir = '/tmp/sudoku-hard/'

    if not os.path.exists(dest_dir):
        print("Downloading data...")

        urllib.request.urlretrieve(url, zip_fname)
        with zipfile.ZipFile(zip_fname) as f:
            f.extractall('/tmp/')

    def read_csv(fname):
        print("Reading %s..." % fname)
        with open(dest_dir + fname) as f:
            reader = csv.reader(f, delimiter=',')
            return [(q, a) for q, a in reader]

    data = read_csv(segment + '.csv')

    def encode(samples):
        def parse(x):
            return list(map(int, list(x)))

        encoded = [(parse(q), parse(a)) for q, a in samples]
        return encoded

    data = encode(data)
    print(f'Number of puzzles in {segment} set : {len(data)}')

    return data


def sudoku_dataloader(batch_size, segment='train', num=-1):
    """
    Get a DataLoader instance for dataset of sudoku. Every iteration of the dataloader returns
    a DGLGraph instance, the ndata of the graph contains:
    'q': question, e.g. the sudoku puzzle to be solved, the position is to be filled with number from 1-9
         if the value in the position is 0
    'a': answer, the ground truth of the sudoku puzzle
    'row': row index for each position in the grid
    'col': column index for each position in the grid
    :param batch_size: Batch size for the dataloader
    :param segment: The segment of the datasets, must in ['train', 'valid', 'test']
    :return: A pytorch DataLoader instance
    """
    data = _get_sudoku_dataset(segment)
    q, a = zip(*data)

    dataset = ListDataset(q, a)
    if segment == 'train':
        if 0 < num < len(dataset):
            print(f'We use {num} out of {len(dataset)} data instances for training.')
            # we first randomly sample num of training data indices
            np.random.seed(0) # fix the random seed to fix the training data
            # randomly shuffle the dataset and get the first num of them
            data_indices = np.arange(len(dataset))
            np.random.shuffle(data_indices)
            data_indices = data_indices[:num]
            dataset = torch.utils.data.Subset(dataset, data_indices)
        data_sampler = RandomSampler(dataset)
    else:
        # for faster evalation, we randomly sample 1k data at most
        if num == -1:
            num = 1000
        num = min(1000, num)
        dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), num, replace=False))
        data_sampler = SequentialSampler(dataset)

    basic_graph = _basic_sudoku_graph()
    sudoku_indices = np.arange(0, 81)
    rows = sudoku_indices // 9
    cols = sudoku_indices % 9

    def collate_fn(batch):
        graph_list = []
        for q, a in batch:
            q = torch.tensor(q, dtype=torch.long)
            a = torch.tensor(a, dtype=torch.long)
            # graph = copy(basic_graph)
            graph = basic_graph.clone()
            graph.ndata['q'] = q  # q means question
            graph.ndata['a'] = a  # a means answer
            graph.ndata['row'] = torch.tensor(rows, dtype=torch.long)
            graph.ndata['col'] = torch.tensor(cols, dtype=torch.long)
            graph_list.append(graph)
        batch_graph = dgl.batch(graph_list)
        return batch_graph

    dataloader = DataLoader(dataset, batch_size, sampler=data_sampler, collate_fn=collate_fn)
    return dataloader


def sudoku_dataloader_semi(batch_size, segment='train', num=-1, batch_label=1):
    """
    Args:
        num: the number of training data
        batch_label: the number of labels in each batch
    Get a DataLoader instance for dataset of sudoku. Every iteration of the dataloader returns
    a DGLGraph instance, the ndata of the graph contains:
    'q': question, e.g. the sudoku puzzle to be solved, the position is to be filled with number from 1-9
         if the value in the position is 0
    'a': answer, the ground truth of the sudoku puzzle
    'row': row index for each position in the grid
    'col': column index for each position in the grid
    :param batch_size: Batch size for the dataloader
    :param segment: The segment of the datasets, must in ['train', 'valid', 'test']
    :return: A pytorch DataLoader instance
    """
    data = _get_sudoku_dataset(segment)
    q, a = zip(*data)

    dataset = ListDataset(q, a)

    if segment == 'train':
        if not 0 < num < len(dataset):
            num = len(dataset)
        print(f'We use {num} out of {len(dataset)} data instances for training.')
        # we first randomly sample num of training data indices
        np.random.seed(0) # fix the random seed to fix the training data
        # randomly shuffle the dataset and get the first num of them
        data_indices = np.arange(len(dataset))
        np.random.shuffle(data_indices)
        data_indices = data_indices[:num]
        # then we split the training data indices into 2 sets for supervised and unsupervised
        idx = int(num * (batch_label / batch_size))
        data_indices_label = data_indices[:idx]
        data_indices_nolabel = data_indices[idx:]
        dataset_label = torch.utils.data.Subset(dataset, data_indices_label)
        dataset_nolabel = torch.utils.data.Subset(dataset, data_indices_nolabel)
        data_sampler_label = RandomSampler(dataset_label)
        data_sampler_nolabel = RandomSampler(dataset_nolabel)
    else:
        data_sampler = SequentialSampler(dataset)

    basic_graph = _basic_sudoku_graph()
    sudoku_indices = np.arange(0, 81)
    rows = sudoku_indices // 9
    cols = sudoku_indices % 9

    def collate_fn(batch):
        graph_list = []
        for q, a in batch:
            q = torch.tensor(q, dtype=torch.long)
            a = torch.tensor(a, dtype=torch.long)
            # graph = copy(basic_graph)
            graph = basic_graph.clone()
            graph.ndata['q'] = q  # q means question
            graph.ndata['a'] = a  # a means answer
            graph.ndata['row'] = torch.tensor(rows, dtype=torch.long)
            graph.ndata['col'] = torch.tensor(cols, dtype=torch.long)
            graph_list.append(graph)
        batch_graph = dgl.batch(graph_list)
        return batch_graph

    if segment == 'train':
        dataloader_label = DataLoader(dataset_label, batch_label, sampler=data_sampler_label, collate_fn=collate_fn)
        dataloader_nolabel = DataLoader(dataset_nolabel, batch_size - batch_label, sampler=data_sampler_nolabel, collate_fn=collate_fn)
        return dataloader_label, dataloader_nolabel
    dataloader = DataLoader(dataset, batch_size, sampler=data_sampler, collate_fn=collate_fn)
    return dataloader