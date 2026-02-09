import os
import torch
import torch.distributed as dist


def run_sequential(rank, size, num_iter=10):
    """
    Prints the process rank sequentially in two orders over `num_iter` iterations,
    separating the output for each iteration by `---`.
    Example (3 processes, num_iter=2):
    ```
    Process 0
    Process 1
    Process 2
    Process 2
    Process 1
    Process 0
    ---
    Process 0
    Process 1
    Process 2
    Process 2
    Process 1
    Process 0
    ```
    """

    buffer = torch.tensor(rank)

    for _ in range(num_iter):
        for i in range(rank):
            dist.recv(buffer, i)
        print(f"Process {rank}")
        sends = []
        for i in range(rank+1, size):
            sends.append(dist.isend(buffer, i))
        for send in sends:
            send.wait()

        for i in range(rank+1, size):
            dist.recv(buffer, i)
        print(f"Process {rank}")
        sends = []
        for i in range(rank):
            sends.append(dist.isend(buffer, i))
        for send in sends:
            send.wait()

        if rank == 0 and _ != num_iter - 1:
            print("---")


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(rank=local_rank, backend="gloo")

    run_sequential(local_rank, dist.get_world_size())
