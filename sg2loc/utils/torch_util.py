"""
Adapted from SceneGraphLoc: https://github.com/y9miao/VLSG.
"""

import random

import numpy as np
import torch
import torch.utils.data


def release_cuda_torch(x):
    """Move all tensors to the CPU, single-element tensors become Python scalars."""
    if isinstance(x, list):
        x = [release_cuda_torch(item) for item in x]
    elif isinstance(x, tuple):
        x = (release_cuda_torch(item) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda_torch(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu()
    return x


def to_cuda(x):
    """Move all tensors to CUDA."""
    if isinstance(x, list):
        x = [to_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (to_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: to_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = x.cuda()
    return x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reset_seed_worker_init_fn(worker_id):
    """Reset the seed for a data loader worker."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(
    dataset,
    batch_size=1,
    num_workers=1,
    shuffle=None,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    distributed=False,
):
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader


def get_test_dataloader(cfg, Dataset):
    """Build the test dataloader where one batch is one query sequence of sequence_length frames."""
    # num_test_data_set=0 pairs every query scan with the first other scan of its room
    test_dataset = Dataset(cfg, split="test", num_test_data_set=0)
    test_dataloader = build_dataloader(
        test_dataset,
        batch_size=cfg.particle_filter.sequence_length,
        num_workers=cfg.val.num_workers,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    return test_dataset, test_dataloader
