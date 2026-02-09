import torch
from syncbn import SyncBatchNorm
from torch.nn.modules.batchnorm import BatchNorm1d
import torch.distributed as dist
import tempfile


def check_bn(rank, size, hid_dim, batch_size, init_file):
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=size,
    )
    torch.manual_seed(0)
    device = torch.device("cuda", rank)
    x = torch.randn(batch_size, hid_dim, requires_grad=True, device=device)

    if rank == 0:
        x.requires_grad_(True)
        bn = BatchNorm1d(hid_dim, affine=False).to(device)
        bn.train()
        golden = bn(x)
        golden_reduced = golden[:batch_size//2].sum()
        golden_reduced.backward()
        golden_grad = x.grad.clone()
    
    syncbn = SyncBatchNorm(hid_dim).to(device)
    syncbn.train()
    my_x = x.detach()[rank::size].clone().requires_grad_(True)
    my_output = syncbn(my_x)

    should_backward = torch.arange(batch_size, device=device)[rank::size] < batch_size//2
    my_output[should_backward].sum().backward()

    if rank == 0:
        for i in range(1, size):
            dist.send(golden_grad, i)
            dist.send(golden, i)
        assert torch.allclose(my_output, golden[rank::size], atol=1e-3, rtol=0)
        assert torch.allclose(my_x.grad, golden_grad[rank::size], atol=1e-3, rtol=0)
    else:
        golden_grad = torch.zeros_like(x)
        golden = torch.zeros_like(x)
        dist.recv(golden_grad, 0)
        dist.recv(golden, 0)
        assert torch.allclose(my_output, golden[rank::size], atol=1e-3, rtol=0)
        assert torch.allclose(my_x.grad, golden_grad[rank::size], atol=1e-3, rtol=0)
    dist.destroy_process_group()


def test_batchnorm(request):
    num_workers = request.config.getoption("--workers")
    hid_dim = request.config.getoption("--hid-dim")
    batch_size = request.config.getoption("--batch-size")
    
    ctx = torch.multiprocessing.get_context("spawn")
    with tempfile.NamedTemporaryFile(delete=False) as f:
        init_file = f.name

    procs = []
    for rank in range(num_workers):
        p = ctx.Process(target=check_bn, args=(rank, num_workers, hid_dim, batch_size, init_file))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        assert p.exitcode == 0
