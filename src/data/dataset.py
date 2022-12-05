""" 
DataSet for the project. 
"""
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm


class SIDDSmall(Dataset):
    """ 
    The dataset for SIDD Small:
    https://www.eecs.yorku.ca/~kamel/sidd/dataset.php
    """

    def __init__(self, root_directory: Path, cache_file: Optional[Path] = None, transform: Optional[torch.nn.Module] = None, dataset_size: int = 160) -> None:
        super().__init__()
        # load the figures
        self.input = {}
        self.target = {}
        self.noise = {}
        self.keys = []
        self.transform = transform
        self.dataset_size = dataset_size
        if cache_file and cache_file.is_file():
            self.load_cache(cache_file)
        else:
            self.load(root_directory, cache_file)

    def load(self, root_directory: Path, cache_file: Optional[Path]):
        """ 
        load the figs in SIDD.
        """
        data_dir = root_directory/"Data"
        all_folders = list(data_dir.iterdir())
        for each in tqdm(all_folders, desc="read sidd data"):
            if each.is_dir():
                key = each.parts[-1]
                for fig in each.iterdir():
                    if "NOISY" in fig.parts[-1]:
                        self.input[key] = read_image(str(fig))
                        self.noise[key] = torch.randn(self.input[key].shape)
                    else:
                        self.target[key] = read_image(str(fig))
        self.keys = list(self.input.keys())

        if cache_file:
            tosave = {
                "input": self.input,
                "target": self.target,
                "keys": self.keys,
                "noise": self.noise
            }
            torch.save(tosave, cache_file)

    def load_cache(self, cache_file: Path):
        """ 
        load cache.
        """
        cache = torch.load(cache_file)
        self.input, self.target, self.keys, self.noise = cache[
            "input"], cache["target"], cache["keys"], cache["noise"]

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        key = self.keys[idx]
        return {
            "key": key,
            "input": self.transform(self.input[key]/255),
            "target": self.transform(self.target[key]/255),
            "noise": self.transform(self.noise[key])
        }

    def __len__(self):
        return min(len(self.keys), self.dataset_size)
