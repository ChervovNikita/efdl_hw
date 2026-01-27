from typing import Optional

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
import random
from copy import deepcopy

import os

MAX_LENGTH = 640

class BrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.samples = []
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        for file in os.listdir(data_path):
            if 'train' not in file:
                continue
            data = open(os.path.join(data_path, file), 'r').readlines()
            for text in tqdm(data[:10000]):
                sample = self.tokenizer(text, padding="max_length", truncation=True, max_length=MAX_LENGTH)
                self.samples.append(torch.tensor(sample['input_ids']).long())

    def __len__(self):
        return len(self.samples)
                
    def __getitem__(self, idx: int):
        return self.samples[idx]


class BigBrainDataset(BrainDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.samples = []
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        for file in os.listdir(data_path):
            if 'train' not in file:
                continue
            data = open(os.path.join(data_path, file), 'r').readlines()
            for text in tqdm(data[:10000]):
                sample = self.tokenizer(text, truncation=True, max_length=MAX_LENGTH)
                self.samples.append(torch.tensor(sample['input_ids']).long())


class UltraBigBrainDataset(BrainDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, k=640):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.samples = []
        self.bins = defaultdict(list)
        for file in os.listdir(data_path):
            if 'train' not in file:
                continue
            data = open(os.path.join(data_path, file), 'r').readlines()
            for text in tqdm(data[:10000]):
                tokens = self.tokenizer(text, truncation=True, max_length=MAX_LENGTH)['input_ids']
                tokens = torch.tensor(tokens).long()
                self.samples.append(tokens)
                self.bins[(tokens.shape[0] - 1) // (k + 1)].append(len(self.samples)-1)


class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        pass

    def __getitem__(self, idx: int):
        pass


def collate_fn(
    batch: list[torch.Tensor], max_length: Optional[int] = MAX_LENGTH
) -> torch.Tensor:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    max_shape = max(b.shape for b in batch)
    result = torch.zeros(len(batch), max_shape[0], dtype=batch[0].dtype)
    for i, b in enumerate(batch):
        result[i, :b.shape[0]] = b
    return result


class UltraBigBrainBatchSampler(Sampler):
    def __init__(self, bins: dict, batch_size: int, max_length: Optional[int] = MAX_LENGTH):
        blocks = []
        for k, v in bins.items():
            blocks.extend([k] * ((len(v) + batch_size - 1) // batch_size))
        self.blocks = blocks
        self.bins_counter = defaultdict(int)
        random.shuffle(blocks)
        self.batch_size = batch_size
        self.bins = deepcopy(bins)
        for k in self.bins.keys():
            random.shuffle(self.bins[k])

    def __len__(self):
        return len(self.blocks)

    def __iter__(self):
        for block in self.blocks:
            yield self.bins[block][self.bins_counter[block]:self.bins_counter[block]+self.batch_size]
            self.bins_counter[block] += self.batch_size
