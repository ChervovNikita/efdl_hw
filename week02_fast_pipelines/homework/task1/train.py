import torch
from torch import nn
from tqdm.auto import tqdm

from unet import Unet

from dataset import get_train_data


class StaticLossScaler:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale_factor

    def step(self, optimizer: torch.optim.Optimizer):
        is_good = True
        with torch.no_grad():
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if not torch.isfinite(param.grad).all():
                        is_good = False
                        break
        if is_good:
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    param.grad /= self.scale_factor
            optimizer.step()


class DynamicLossScaler:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
        self.without_overflow_count = 0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale_factor

    def step(self, optimizer: torch.optim.Optimizer):
        is_good = True
        with torch.no_grad():
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if not torch.isfinite(param.grad).all():
                        is_good = False
                        break
        if is_good:
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    param.grad /= self.scale_factor
            optimizer.step()
            self.without_overflow_count += 1
            if self.without_overflow_count > 5:
                self.scale_factor *= 2
                self.without_overflow_count = 0
        else:
            self.without_overflow_count = 0
            self.scale_factor /= 2


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaling_type: str,
) -> None:
    model.train()

    if scaling_type == 'static':
        scaler = StaticLossScaler(scale_factor=1024)
    elif scaling_type == 'dynamic':
        scaler = DynamicLossScaler(scale_factor=16384)
    elif scaling_type == 'torch':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        if scaler is not None:
            optimizer.zero_grad()
            with torch.amp.autocast(device.type, dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            if scaling_type == 'torch':
                scaler.update()
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train(scaling_type):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()

    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, device=device, scaling_type=scaling_type)


# if __name__ == "__main__":
#     train()
