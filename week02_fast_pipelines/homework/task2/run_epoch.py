from enum import Enum
from transformer import PositionalEncoding, generate_square_subsequent_mask
from torch import nn
from dataset import BrainDataset, BigBrainDataset, UltraBigBrainDataset, UltraDuperBigBrainDatasetNaive, UltraDuperBigBrainDatasetFFD, UltraDuperBigBrainDatasetOBFD, collate_fn, UltraBigBrainBatchSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import math
import time
import pickle
from copy import deepcopy

import torch


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_BIG_BRAIN = 3
    ULTRA_DUPER_BIG_BRAIN_NAIVE = 4
    ULTRA_DUPER_BIG_BRAIN_FFD = 5
    ULTRA_DUPER_BIG_BRAIN_OBFD = 6


class GPT2LikeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 1024
        self.d_hid = 1024
        self.num_heads = 8
        self.tokenizer_vocab = 30522
        self.embedding = torch.nn.Embedding(self.tokenizer_vocab, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(self.d_model, self.num_heads, self.d_hid, batch_first=True),
            num_layers=1,
        )
        self.linear = nn.Linear(self.d_hid, self.tokenizer_vocab)

    def forward(self, x: torch.Tensor, seq_mask=None) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        memory = torch.zeros_like(x)
        x = self.transformer_decoder(x, memory, tgt_mask=seq_mask)
        x = self.linear(x)
        return x

def get_gpt2_model() -> torch.nn.Module:
    return GPT2LikeModel()


def run_epoch(data_mode: DataMode, k=None) -> None:
    model = get_gpt2_model()
    model.to('cuda')
    model.train()

    if data_mode == DataMode.BRAIN:
        dataset = BrainDataset(data_path='wikitext-103-raw-v1')
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    elif data_mode == DataMode.BIG_BRAIN:
        dataset = BigBrainDataset(data_path='wikitext-103-raw-v1')
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    elif data_mode == DataMode.ULTRA_BIG_BRAIN:
        dataset = UltraBigBrainDataset(data_path='wikitext-103-raw-v1', k=k)
        bins = dataset.bins
        batch_sampler = UltraBigBrainBatchSampler(bins, batch_size=16)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
    elif data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN_NAIVE:
        dataset = UltraDuperBigBrainDatasetNaive(data_path='wikitext-103-raw-v1')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    elif data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN_FFD:
        dataset = UltraDuperBigBrainDatasetFFD(data_path='wikitext-103-raw-v1')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    elif data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN_OBFD:
        dataset = UltraDuperBigBrainDatasetOBFD(data_path='wikitext-103-raw-v1')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    else:
        assert False

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    scaler = torch.cuda.amp.GradScaler()

    # i = 0
    losses = []
    loop = tqdm(dataloader, total=len(dataloader))
    times = []
    for batch in loop:
        start_time = time.time()
        if data_mode not in [DataMode.ULTRA_DUPER_BIG_BRAIN_NAIVE, DataMode.ULTRA_DUPER_BIG_BRAIN_FFD, DataMode.ULTRA_DUPER_BIG_BRAIN_OBFD]:
            input_ids = batch.to('cuda')
            seq_mask = generate_square_subsequent_mask(input_ids.shape[0])
        else:
            input_ids = batch['input_ids'].to('cuda')
            seq_mask = batch['seq_mask'].to('cuda')[0]

        input_ids = input_ids.to('cuda')
        optimizer.zero_grad()
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                if data_mode not in [DataMode.ULTRA_DUPER_BIG_BRAIN_NAIVE, DataMode.ULTRA_DUPER_BIG_BRAIN_FFD, DataMode.ULTRA_DUPER_BIG_BRAIN_OBFD]:
                    outputs = model(input_ids, None)
                else:
                    outputs = model(input_ids, seq_mask)
                output_plain = outputs[:, :-1, :].reshape(-1, outputs.shape[-1])
                # target_plain = input_ids[:, 1:].reshape(-1)
                # is_pad_mask = (target_plain == tokenizer.pad_token_id)
                # loss = criterion(output_plain[~is_pad_mask], target_plain[~is_pad_mask])
                # losses.append(loss.item())
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # i += 1
        # if i % 100 == 0:
        #     loop.set_description(f"Loss: {round(sum(losses) / len(losses), 4)}")
        #     losses = []
        torch.cuda.synchronize()
        times.append(time.time() - start_time)
    path = f'{data_mode.name}'
    if k is not None:
        path += f'_k{k}'
    path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump({'times': times}, f)


if __name__ == "__main__":
    run_epoch(DataMode.ULTRA_DUPER_BIG_BRAIN_FFD)
