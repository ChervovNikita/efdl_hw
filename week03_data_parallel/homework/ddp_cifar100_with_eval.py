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
from torch.utils.data import Subset

torch.set_num_threads(1)


def init_process(local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


class Net(nn.Module):
    """
    A very simple model with minimal changes from the tutorial, used for the sake of simplicity.
    Feel free to replace it with EffNetV2-XL once you get comfortable injecting SyncBN into models programmatically.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 100)
        # self.bn1 = nn.BatchNorm1d(128, affine=False)  # to be replaced with SyncBatchNorm
        self.bn1 = SyncBatchNorm(128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def run_training(rank, size):
    torch.manual_seed(0)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    dataset = CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        download=True,
    )
    # where's the validation dataset?
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset, size, rank), batch_size=64)
    
    if rank == 0:
        val_ids = torch.tensor(list(range(val_size)), device=device, dtype=torch.long)[:val_size//size*size].reshape(size, -1)
        val_ids_list = [val_ids[i] for i in range(size)]
    else:
        val_ids_list = None
    my_ids = torch.zeros(val_size//size*size, device=device, dtype=torch.long)
    dist.scatter(my_ids, val_ids_list, src=0)
    val_dataset = Subset(val_dataset, my_ids.tolist())
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = Net()
    # device = torch.device("cuda")  # replace with "cuda" afterwards
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = len(loader)

    accumulate_steps = 2

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=False,
        with_stack=True,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
    ) as prof:
        for _ in range(10):
            accuracies = []
            epoch_loss = torch.zeros((1,), device=device)
            
            i = 0
            for step, (data, target) in enumerate(loader):
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                epoch_loss += (loss / accumulate_steps).detach()
                loss.backward()

                if (step + 1) % accumulate_steps == 0 or step == num_batches - 1:
                    average_gradients(model)
                    optimizer.step()
                    optimizer.zero_grad()

                acc = (output.argmax(dim=1) == target).float().mean()
                accuracies.append(acc.detach())
                epoch_loss = 0
                prof.step()
            if rank == 0:
                print(f"accuracy: {torch.mean(torch.stack(accuracies))}")
            
            eval_accuracies = torch.zeros(len(val_loader), device=device)
            eval_losses = torch.zeros(len(val_loader), device=device)

            with torch.no_grad():
                for idx, (data, target) in enumerate(val_loader):
                    data = data.to(device)
                    target = target.to(device)

                    output = model(data)
                    loss = torch.nn.functional.cross_entropy(output, target, reduction='none')
                    eval_losses[idx] = loss.detach().mean()
                    acc = (output.argmax(dim=1) == target).float().mean()
                    eval_accuracies[idx] = acc.detach()

            if rank != 0:
                dist.send(eval_accuracies, 0)
                dist.send(eval_losses, 0)
            else:
                buffer_accuracies = torch.zeros(len(eval_accuracies), device=device)
                buffer_losses = torch.zeros(len(eval_losses), device=device)
                for i in range(1, size):
                    dist.recv(buffer_accuracies, i)
                    dist.recv(buffer_losses, i)
                    eval_accuracies += buffer_accuracies
                    eval_losses += buffer_losses
                eval_accuracy = (eval_accuracies / size).mean()
                eval_loss = (eval_losses / size).mean()
                print(f"eval_accuracy: {eval_accuracy}, eval_loss: {eval_loss}")

    torch.cuda.synchronize()
    end = time.time()
    print(f"Time: {end - start}")
    print("Max allocated", torch.cuda.max_memory_allocated() / 1024**2)
    prof.export_chrome_trace(f"trace_my_rank{rank}.json")


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_training, backend="nccl")  # replace with "nccl" when testing on several GPUs
