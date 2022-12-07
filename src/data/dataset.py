""" 
DataSet for the project. 
"""
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class SIDDSmall(Dataset):
    """ 
    The dataset for SIDD Small:
    https://www.eecs.yorku.ca/~kamel/sidd/dataset.php
    wrap only a single dir
    """

    def __init__(self, img_directory: Path, transform: Optional[torch.nn.Module] = None, batch_size: int = 1) -> None:
        super().__init__()
        # load the figures
        for fig in img_directory.iterdir():
            if "NOISY" in fig.parts[-1]:
                self.noise_fig = read_image(str(fig))
            else:
                self.target_fig = read_image(str(fig))
        self.noise_input = torch.randn(self.noise_fig.shape)
        self.key = img_directory.parts[-1]
        self.transform = transform
        self.batch_size = batch_size

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        # only one item in the dataset
        return {
            "key": self.key,
            "noise_fig": self.transform(self.noise_fig/255),
            "target_fig": self.transform(self.target_fig/255),
            "noise_input": self.transform(self.noise_input)
        }

    def __len__(self):
        return self.batch_size


class SIDDRandomMask(Dataset):
    """ 
    The dataset modified from SIDD Small:
    https://www.eecs.yorku.ca/~kamel/sidd/dataset.php
    wrap only a single dir
    randomly mask out the images
    """

    def __init__(self, img_directory: Path, transform: Optional[torch.nn.Module] = None, batch_size: int = 1, mask_threshold=0.5, image_resize: List[int] = None) -> None:
        super().__init__()
        # load the figures
        for fig in img_directory.iterdir():
            if "NOISY" in fig.parts[-1]:
                continue
            else:
                self.target_fig = read_image(str(fig))
                # generate mask
                self.mask = torch.rand(
                    [self.target_fig.shape[0]]+list(image_resize))
                self.mask[self.mask < mask_threshold] = 0
                self.mask[self.mask != 0] = 1

        self.noise_input = torch.randn(self.target_fig.shape)
        self.key = img_directory.parts[-1]
        self.transform = transform
        self.batch_size = batch_size

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        # only one item in the dataset
        return {
            "key": self.key,
            "mask_fig": self.transform(self.target_fig/255)*self.mask,
            "target_fig": self.transform(self.target_fig/255),
            "noise_input": self.transform(self.noise_input),
            "mask": self.mask
        }

    def __len__(self):
        return self.batch_size
