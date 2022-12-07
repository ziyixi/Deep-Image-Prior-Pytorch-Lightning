""" 
pytorch lightning core data module.
"""
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize

from src.conf import DataConfig
from src.data.dataset import SIDDRandomMask, SIDDSmall


class DeepImagePriorDataModule(pl.LightningDataModule):
    """ 
    the core pytorch lightning data module.
    Due to the specific structure, the test and the train data loader is the same.
    """

    def __init__(self, data_conf: DataConfig, img_dir: Path) -> None:
        super().__init__()
        self.data_conf = data_conf
        self.dataset: Optional[Dataset] = None
        self.img_dir = img_dir

    def setup(self, stage: Optional[str] = None) -> None:
        # only one dataset for train/test, no need to check stage.
        transform = Compose([Resize(tuple(self.data_conf.image_resize))])
        if self.data_conf.data_type == "sidd_small":
            self.dataset = SIDDSmall(
                self.img_dir, transform=transform, batch_size=self.data_conf.batch_size)
        elif self.data_conf.data_type == "sidd_mask":
            self.dataset = SIDDRandomMask(
                self.img_dir, transform=transform, batch_size=self.data_conf.batch_size, mask_threshold=self.data_conf.mask_threshold, image_resize=self.data_conf.image_resize)
        else:
            raise Exception(
                f"data type {self.data_conf.data_type} is not implemented!")

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.data_conf.batch_size, shuffle=False, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.data_conf.batch_size, shuffle=False, num_workers=0)
