""" 
DataSet for the project. 
"""
from pathlib import Path
from typing import Dict, Optional, Union

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
