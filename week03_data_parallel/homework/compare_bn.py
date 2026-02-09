import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR100
from syncbn import SyncBatchNorm
import time
from torch.profiler import profile, ProfilerActivity

torch.set_num_threads(1)


def init_process(local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


def run_compare_bn(rank, size):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(0)
    for step in range(2):
        for hid_dim in [128, 256, 512, 1024]:
            for batch_size in [32, 64]:
                dist.barrier()
                torch.cuda.reset_peak_memory_stats()
                start = time.time()
                for _ in range(100):
                    x = torch.randn(batch_size // size, hid_dim, requires_grad=True, device=device)
                    bn = SyncBatchNorm(hid_dim).to(device)
                    bn.train()
                    y = bn(x)
                    y.sum().backward()
                torch.cuda.synchronize()
                end = time.time()
                max_memory = torch.cuda.max_memory_allocated() / 1024**2
                dist.barrier()
                if rank == 0 and step != 0:
                    print(f"my bn, for hidden {hid_dim} and batch size {batch_size} Time: {end - start}, Max memory: {max_memory}")
                dist.barrier()

    torch.manual_seed(0)
    for step in range(2):
        for hid_dim in [128, 256, 512, 1024]:
            for batch_size in [32, 64]:
                dist.barrier()
                torch.cuda.reset_peak_memory_stats()
                start = time.time()
                for _ in range(100):
                    x = torch.randn(batch_size // size, hid_dim, requires_grad=True, device=device)
                    bn = nn.SyncBatchNorm(hid_dim).to(device)
                    bn.train()
                    y = bn(x)
                    y.sum().backward()
                torch.cuda.synchronize()
                end = time.time()
                max_memory = torch.cuda.max_memory_allocated() / 1024**2
                dist.barrier()
                if rank == 0 and step != 0:
                    print(f"torch bn, for hidden {hid_dim} and batch size {batch_size} Time: {end - start}, Max memory: {max_memory}")
                dist.barrier()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_compare_bn, backend="nccl")  # replace with "nccl" when testing on several GPUs
