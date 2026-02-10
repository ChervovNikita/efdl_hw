import os
import random

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import time
import time
import psutil


def monitor_system_ram_while(processes, interval=0.05):  # thx chatgpt for monitoring
    """
    Sample total system RAM until all processes are done.
    Returns (baseline_used_bytes, peak_used_bytes).
    """
    baseline = psutil.virtual_memory().used
    peak = baseline
    t0 = time.time()

    while True:
        used = psutil.virtual_memory().used
        if used > peak:
            peak = used

        alive = any(p.is_alive() for p in processes)
        if not alive:
            break

        time.sleep(interval)

    return baseline, peak

def _mb(x): 
    return x / (1024**2)


def init_process(rank, size, world_size, verbose, fn, master_port, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(rank, size, world_size, verbose)


def butterfly_allreduce(send, rank, size):
    """
    Performs Butterfly All-Reduce over the process group. Modifies the input tensor in place.
    Args:
        send: torch.Tensor to be averaged with other processes.
        rank: Current process rank (in a range from 0 to size)
        size: Number of workers
    """

    buffer_for_chunk = torch.empty((size,), dtype=torch.float)

    send_futures = []

    for i, elem in enumerate(send):
        if i != rank:
            send_futures.append(dist.isend(elem, i))

    recv_futures = []

    for i, elem in enumerate(buffer_for_chunk):
        if i != rank:
            recv_futures.append(dist.irecv(elem, i))
        else:
            elem.copy_(send[i])

    for future in recv_futures:
        future.wait()

    # compute the average
    torch.mean(buffer_for_chunk, dim=0, out=send[rank])

    for i in range(size):
        if i != rank:
            send_futures.append(dist.isend(send[rank], i))

    recv_futures = []

    for i, elem in enumerate(send):
        if i != rank:
            recv_futures.append(dist.irecv(elem, i))

    for future in recv_futures:
        future.wait()
    for future in send_futures:
        future.wait()


def get_shape(total_shape, idx, size):
    total = total_shape[0]
    base = total // size
    if idx < total % size:
        base += 1
    return tuple([base] + [s for s in total_shape[1:]])

def get_begin_end(total_shape, idx, size):
    first_extra = total_shape[0] % size
    base = total_shape[0] // size
    if idx < first_extra:
        start = idx * (base + 1)
        end = start + base + 1
    else:
        start = first_extra * (base + 1) + (idx - first_extra) * base
        end = start + base
    return start, end


def ring_allreduce(send, rank, size):
    """
    Performs Ring All-Reduce over the process group. Modifies the input tensor in place.
    Args:
        send: torch.Tensor to be averaged with other processes.
        rank: Current process rank (in a range from 0 to size)
        size: Number of workers
    """
    forward_best = rank
    backward_best = rank
    for _ in range(size-1):
        reqs = []

        get_forward = (forward_best - 1 + size) % size
        if get_shape(send.shape, get_forward, size*2)[0] != 0:
            buffer_forward = torch.zeros(get_shape(send.shape, get_forward, size*2))
            reqs.append(dist.irecv(buffer_forward, (rank-1+size)%size))
        
        get_backward = (backward_best + 1 + size) % size
        if get_shape(send.shape, get_backward+size, size*2)[0] != 0:
            buffer_backward = torch.zeros(get_shape(send.shape, get_backward+size, size*2))
            reqs.append(dist.irecv(buffer_backward, (rank+1)%size))

        if get_shape(send.shape, forward_best, size*2)[0] != 0:
            start, end = get_begin_end(send.shape, forward_best, size*2)
            reqs.append(dist.isend(send[start:end], (rank+1)%size))

        if get_shape(send.shape, backward_best+size, size*2)[0] != 0:
            start, end = get_begin_end(send.shape, backward_best+size, size*2)
            reqs.append(dist.isend(send[start:end], (rank-1+size)%size))

        for req in reqs:
            req.wait()

        if get_shape(send.shape, get_forward, size*2)[0] != 0:
            start, end = get_begin_end(send.shape, get_forward, size*2)
            send[start:end].add_(buffer_forward)
        if get_shape(send.shape, get_backward+size, size*2)[0] != 0:
            start, end = get_begin_end(send.shape, get_backward+size, size*2)
            send[start:end].add_(buffer_backward)
        forward_best = get_forward
        backward_best = get_backward
    
    start, end = get_begin_end(send.shape, forward_best, size*2)
    send[start:end].mul_(1/size)
    start, end = get_begin_end(send.shape, backward_best+size, size*2)
    send[start:end].mul_(1/size)

    for _ in range(size-1):
        reqs = []
        get_forward = (forward_best - 1 + size) % size

        if get_shape(send.shape, get_forward, size*2)[0] != 0:
            buffer_forward = torch.zeros(get_shape(send.shape, get_forward, size*2))
            reqs.append(dist.irecv(buffer_forward, (rank-1+size)%size))
        
        get_backward = (backward_best + 1 + size) % size
        if get_shape(send.shape, get_backward+size, size*2)[0] != 0:
            buffer_backward = torch.zeros(get_shape(send.shape, get_backward+size, size*2))
            reqs.append(dist.irecv(buffer_backward, (rank+1)%size))

        if get_shape(send.shape, forward_best, size*2)[0] != 0:
            start, end = get_begin_end(send.shape, forward_best, size*2)
            reqs.append(dist.isend(send[start:end], (rank+1)%size))

        if get_shape(send.shape, backward_best+size, size*2)[0] != 0:
            start, end = get_begin_end(send.shape, backward_best+size, size*2)
            reqs.append(dist.isend(send[start:end], (rank-1+size)%size))

        for req in reqs:
            req.wait()

        if get_shape(send.shape, get_forward, size*2)[0] != 0:
            start, end = get_begin_end(send.shape, get_forward, size*2)
            send[start:end].copy_(buffer_forward)
        if get_shape(send.shape, get_backward+size, size*2)[0] != 0:
            start, end = get_begin_end(send.shape, get_backward+size, size*2)
            send[start:end].copy_(buffer_backward)
        forward_best = get_forward
        backward_best = get_backward


def run_butterfly_allreduce(rank, size, world_size, verbose=False):
    """Simple point-to-point communication."""
    torch.manual_seed(rank)
    tensor = torch.randn((size,), dtype=torch.float)
    # if verbose:
    #     print("Rank ", rank, " has data ", tensor)
    butterfly_allreduce(tensor, rank, world_size)
    if verbose:
        print(tensor)


def run_ring_allreduce(rank, size, world_size, verbose=False):
    """Simple point-to-point communication."""
    torch.manual_seed(rank)
    tensor = torch.randn((size,), dtype=torch.float)
    # if verbose:
    #     print("Rank ", rank, " has data ", tensor)
    ring_allreduce(tensor, rank, world_size)
    if verbose:
        print(tensor)


def run_torch_allreduce(rank, size, world_size, verbose=False):
    """Simple point-to-point communication."""
    torch.manual_seed(rank)
    tensor = torch.randn((size,), dtype=torch.float)
    # if verbose:
    #     print("Rank ", rank, " has data ", tensor)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(world_size)
    if verbose:
        print(tensor)


def bench(world_size, size, verbose, should_butterfly=True):
    processes = []
    port = random.randint(25000, 30000)
    
    if should_butterfly:
        print('butterfly')
        start = time.time()
        for rank in range(world_size):
            p = Process(target=init_process, args=(rank, size, world_size, verbose, run_butterfly_allreduce, port))
            p.start()
            processes.append(p)

        pids = [p.pid for p in processes]
        baseline, peak = monitor_system_ram_while(processes)
        print(f'peak ram {_mb(peak - baseline)} MB')
        for p in processes: 
            p.join()
        end = time.time()
        print(f'Time: {end - start}')
    
    print('ring')
    processes = []
    start = time.time()
    for rank in range(world_size):
        p = Process(target=init_process, args=(rank, size, world_size, verbose, run_ring_allreduce, port))
        p.start()
        processes.append(p)

    pids = [p.pid for p in processes]
    baseline, peak = monitor_system_ram_while(processes)
    print(f'peak ram {_mb(peak - baseline)} MB')
    for p in processes:
        p.join()
    end = time.time()
    print(f'Time: {end - start}')

    print('torch')
    processes = []
    start = time.time()
    for rank in range(world_size):
        p = Process(target=init_process, args=(rank, size, world_size, verbose, run_torch_allreduce, port))
        p.start()
        processes.append(p)

    pids = [p.pid for p in processes]
    baseline, peak = monitor_system_ram_while(processes)
    print(f'peak ram {_mb(peak - baseline)} MB')
    for p in processes:
        p.join()
    end = time.time()
    print(f'Time: {end - start}')

if __name__ == "__main__":
    world_size = 8
    size = 8
    print('8, 8')
    bench(world_size, size, True)

    print()
    print('-' * 100)
    print()

    world_size = 4
    size = 7
    print('4, 7')
    bench(world_size, size, True, should_butterfly=False)

    print()
    print('-' * 100)
    print()

    world_size = 32
    size = 32
    print('32, 32')
    bench(world_size, size, False, should_butterfly=True)

    print()
    print('-' * 100)
    print()

    world_size = 16
    size = 10000
    print('16, 10000')
    bench(world_size, size, False, should_butterfly=False)
