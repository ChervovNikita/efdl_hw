import typing as tp

import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import pandas as pd

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import Settings, Clothes, seed_everything
from vit import ViT
from profiler import Profile


def get_vit_model() -> torch.nn.Module:
    model = ViT(
        depth=12,
        heads=4,
        image_size=224,
        patch_size=32,
        num_classes=20,
        channels=3,
    ).to(Settings.device)
    return model


def get_loaders() -> torch.utils.data.DataLoader:
    # dataset.download_extract_dataset()
    train_transforms = dataset.get_train_transforms()
    val_transforms = dataset.get_val_transforms()

    frame = pd.read_csv(f"{Clothes.directory}/{Clothes.csv_name}")
    train_frame = frame.sample(frac=Settings.train_frac)
    val_frame = frame.drop(train_frame.index)

    train_data = dataset.ClothesDataset(
        f"{Clothes.directory}/{Clothes.train_val_img_dir}", train_frame, transform=train_transforms
    )
    val_data = dataset.ClothesDataset(
        f"{Clothes.directory}/{Clothes.train_val_img_dir}", val_frame, transform=val_transforms
    )

    print(f"Train Data: {len(train_data)}")
    print(f"Val Data: {len(val_data)}")

    train_loader = DataLoader(dataset=train_data, batch_size=Settings.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(dataset=val_data, batch_size=Settings.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return train_loader, val_loader


def run_epoch(model, train_loader, val_loader, criterion, optimizer, profiler_type) -> tp.Tuple[float, float]:
    epoch_loss, epoch_accuracy = torch.tensor(0.0, device=Settings.device), torch.tensor(0.0, device=Settings.device)
    val_loss, val_accuracy = torch.tensor(0.0, device=Settings.device), torch.tensor(0.0, device=Settings.device)
    model.train()
    if profiler_type == 'my':
        profiler = Profile(model, name="train", wait=1, warmup=2, active=3, repeat=2)
    else:
        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=2),
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
        )

    scaler = torch.cuda.amp.GradScaler()
    
    with profiler as prof:
        for data, label in tqdm(train_loader, desc="Train"):
            data = data.to(Settings.device, non_blocking=True)
            label = label.to(Settings.device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                output = model(data)
                loss = criterion(output, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc.detach()
            epoch_loss += loss.detach()
            prof.step()
    epoch_accuracy = epoch_accuracy.detach().cpu().item() / len(train_loader)
    epoch_loss = epoch_loss.detach().cpu().item() / len(train_loader)

    torch.cuda.synchronize()
    
    if profiler_type == 'my':
        prof.to_perfetto("train.json")
    else:
        prof.export_chrome_trace("train_torch.json")

    model.eval()

    if profiler_type == 'my':
        profiler = Profile(model, name="val", wait=1, warmup=1, active=3, repeat=1)
    else:
        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
        )

    with profiler as prof:
        with torch.no_grad():
            for data, label in tqdm(val_loader, desc="Val"):
                data = data.to(Settings.device, non_blocking=True)
                label = label.to(Settings.device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    output = model(data)
                    loss = criterion(output, label)
                acc = (output.argmax(dim=1) == label).float().mean()
                val_accuracy += acc.detach()
                val_loss += loss.detach().item()
                prof.step()
    
    val_accuracy = val_accuracy.detach().cpu().item() / len(val_loader)
    val_loss = val_loss.detach().cpu().item() / len(val_loader)

    torch.cuda.synchronize()
    if profiler_type == 'my':
        prof.to_perfetto("val.json")
    else:
        prof.export_chrome_trace("val_torch.json")

    return epoch_loss, epoch_accuracy, val_loss, val_accuracy


def main(profiler_type):
    seed_everything()
    model = get_vit_model()
    train_loader, val_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Settings.lr)
    run_epoch(model, train_loader, val_loader, criterion, optimizer, profiler_type)
    return


if __name__ == "__main__":
    main(profiler_type='my')
