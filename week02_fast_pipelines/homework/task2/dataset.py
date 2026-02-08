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
            for text in tqdm(data):
                sample = self.tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
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
            for text in tqdm(data):
                sample = self.tokenizer(text, truncation=True, max_length=max_length)
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
            for text in tqdm(data):
                tokens = self.tokenizer(text, truncation=True, max_length=max_length)['input_ids']
                tokens = torch.tensor(tokens).long()
                self.samples.append(tokens)
                self.bins[(tokens.shape[0] - 1) // (k + 1)].append(len(self.samples)-1)


class UltraDuperBigBrainDatasetNaive(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.raw_samples = []
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        for file in os.listdir(data_path):
            if 'train' not in file:
                continue
            data = open(os.path.join(data_path, file), 'r').readlines()
            for text in tqdm(data):
                sample = self.tokenizer(text, truncation=True, max_length=max_length)
                self.raw_samples.append(torch.tensor(sample['input_ids']).long())
        self.final_samples = [torch.full((max_length,), self.tokenizer.pad_token_id, dtype=torch.long)]
        self.shapes = [[]]
        last_len = 0
        for sample in self.raw_samples:
            if last_len + sample.shape[0] <= max_length:
                self.final_samples[-1][last_len:last_len+sample.shape[0]] = sample
                last_len += sample.shape[0]
                self.shapes[-1].append(sample.shape[0])
            else:
                keep_first = max_length - last_len
                self.final_samples[-1][last_len:last_len+keep_first] = sample[:keep_first]
                self.shapes[-1].append(keep_first)
                self.final_samples.append(torch.full((max_length,), self.tokenizer.pad_token_id, dtype=torch.long))
                self.shapes.append([])
                self.final_samples[-1][:sample.shape[0]-keep_first] = sample[keep_first:]
                last_len = sample.shape[0]-keep_first
                self.shapes[-1].append(sample.shape[0]-keep_first)
        # self.seq_masks = []
        # for shapes in self.shapes:
        #     seq_mask = torch.ones(max_length, max_length) * float("-inf")
        #     cur_begin = 0
        #     for shape in shapes:
        #         cur_end = cur_begin + shape
        #         seq_mask[cur_begin:cur_end, cur_begin:cur_end] = torch.triu(torch.ones(shape, shape) * float("-inf"), diagonal=1)
        #         cur_begin = cur_end
        #     seq_mask[torch.arange(max_length), torch.arange(max_length)] = 0
        #     self.seq_masks.append(torch.tensor(seq_mask))

    def __len__(self):
        return len(self.final_samples)

    
    def __getitem__(self, idx: int):
        shapes = self.shapes[idx]
        seq_mask = torch.ones(self.max_length, self.max_length) * float("-inf")
        cur_begin = 0
        for shape in shapes:    
            cur_end = cur_begin + shape
            seq_mask[cur_begin:cur_end, cur_begin:cur_end] = torch.triu(torch.ones(shape, shape) * float("-inf"), diagonal=1)
            cur_begin = cur_end
        seq_mask[torch.arange(self.max_length), torch.arange(self.max_length)] = 0
        return {
            'input_ids': self.final_samples[idx],
            'seq_mask': torch.tensor(seq_mask)
        }


class UltraDuperBigBrainDatasetFFD(UltraDuperBigBrainDatasetNaive):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.raw_samples = []
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        for file in os.listdir(data_path):
            if 'train' not in file:
                continue
            data = open(os.path.join(data_path, file), 'r').readlines()
            for text in tqdm(data):
                sample = self.tokenizer(text, truncation=True, max_length=max_length+1)
                sample = torch.tensor(sample['input_ids']).long()
                if sample.shape[0] > max_length:
                    continue
                self.raw_samples.append(sample[:max_length])
        self.raw_samples.sort(key=lambda x: x.shape[0], reverse=True)
        self.final_samples = []
        self.shapes = []
        current_lens = []
        current_lens_torch = torch.zeros(len(self.raw_samples))
        for sample in tqdm(self.raw_samples):
            i = (current_lens_torch[:len(self.final_samples)+1] + sample.shape[0] <= max_length).int().argmax().item()
            if i < len(self.final_samples):
                self.shapes[i].append(sample.shape[0])
                self.final_samples[i][current_lens[i]:current_lens[i]+sample.shape[0]] = sample
                current_lens[i] += sample.shape[0]
                current_lens_torch[i] += sample.shape[0]
            else:
                self.final_samples.append(torch.full((max_length,), self.tokenizer.pad_token_id, dtype=torch.long))
                self.final_samples[-1][:sample.shape[0]] = sample
                current_lens.append(sample.shape[0])
                current_lens_torch[len(current_lens)-1] = sample.shape[0]
                self.shapes.append([sample.shape[0]])

        # self.seq_masks = []
        # for shapes in tqdm(self.shapes):
        #     seq_mask = torch.ones(max_length, max_length) * float("-inf")
        #     cur_begin = 0
        #     for shape in shapes:
        #         cur_end = cur_begin + shape
        #         seq_mask[cur_begin:cur_end, cur_begin:cur_end] = torch.triu(torch.ones(shape, shape) * float("-inf"), diagonal=1)
        #         cur_begin = cur_end
        #     seq_mask[torch.arange(max_length), torch.arange(max_length)] = 0
        #     self.seq_masks.append(torch.tensor(seq_mask))


class STNode:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.mid = (start + end) // 2
        if self.start == self.end:
            self.left = None
            self.right = None
        else:
            self.left = STNode(start, self.mid)
            self.right = STNode(self.mid+1, end)
        self.value = 0


def get_first(root: STNode, length: int):
    if root.start == root.end:
        if root.value:
            return root.start
        else:
            return None
    if length <= root.mid and root.left.value:
        result = get_first(root.left, length)
        if result is not None:
            return result
    return get_first(root.right, length)


def update(root: STNode, v: int, counter: int):
    if root.start == root.end:
        root.value = counter
        return
    if v <= root.mid:
        update(root.left, v, counter)
    else:
        update(root.right, v, counter)
    root.value = root.left.value + root.right.value


class UltraDuperBigBrainDatasetOBFD(UltraDuperBigBrainDatasetNaive):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.raw_samples = []
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        for file in os.listdir(data_path):
            if 'train' not in file:
                continue
            data = open(os.path.join(data_path, file), 'r').readlines()
            for text in tqdm(data):
                sample = self.tokenizer(text, truncation=True, max_length=max_length+1)
                sample = torch.tensor(sample['input_ids']).long()
                if sample.shape[0] > max_length:
                    continue
                self.raw_samples.append(sample[:max_length])
        self.raw_samples.sort(key=lambda x: x.shape[0], reverse=True)

        start = 1
        while start < max_length + 1:
            start *= 2
        self.root = STNode(0, start-1)

        size_to_ids = defaultdict(list)

        self.final_samples = []
        self.shapes = []
        current_lens = []
        for sample in tqdm(self.raw_samples):
            first_free = get_first(self.root, sample.shape[0])
            if first_free is None:
                self.final_samples.append(torch.full((max_length,), self.tokenizer.pad_token_id, dtype=torch.long))
                self.final_samples[-1][:sample.shape[0]] = sample
                current_lens.append(sample.shape[0])
                self.shapes.append([sample.shape[0]])
                size_to_ids[max_length-sample.shape[0]].append(len(self.final_samples)-1)
                update(self.root, max_length-sample.shape[0], len(size_to_ids[max_length-sample.shape[0]]))
                continue
            i = size_to_ids[first_free].pop()
            self.shapes[i].append(sample.shape[0])
            self.final_samples[i][current_lens[i]:current_lens[i]+sample.shape[0]] = sample
            current_lens[i] += sample.shape[0]
            update(self.root, first_free, len(size_to_ids[first_free]))

            size_to_ids[max_length-current_lens[i]].append(i)
            update(self.root, max_length-current_lens[i], len(size_to_ids[max_length-current_lens[i]]))

        # self.seq_masks = []
        # for shapes in self.shapes:
        #     seq_mask = torch.ones(max_length, max_length) * float("-inf")
        #     cur_begin = 0
        #     for shape in shapes:
        #         cur_end = cur_begin + shape
        #         seq_mask[cur_begin:cur_end, cur_begin:cur_end] = torch.triu(torch.ones(shape, shape) * float("-inf"), diagonal=1)
        #         cur_begin = cur_end
        #     seq_mask[torch.arange(max_length), torch.arange(max_length)] = 0
        #     self.seq_masks.append(torch.tensor(seq_mask))


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
