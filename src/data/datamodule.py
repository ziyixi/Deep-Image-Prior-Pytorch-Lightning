""" 
pytorch lightning core data module.
"""
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize

from src.conf import DataConfig
from src.data.dataset import SIDDSmall


class DeepImagePriorDataModule(pl.LightningDataModule):
    """ 
    the core pytorch lightning data module.
    Due to the specific structure, the test and the train data loader is the same.
    """

    def __init__(self, data_conf: DataConfig) -> None:
        super().__init__()
        self.data_conf = data_conf
        self.dataset: Optional[Dataset] = None

    def prepare_data(self):
        # cache
        if self.data_conf.cache_file.is_file():
            return
        if self.data_conf.data_type == "sidd_small":
            SIDDSmall(self.data_conf.root_dir, self.data_conf.cache_file)
        else:
            raise Exception("not implemented yet!")

    def setup(self, stage: Optional[str] = None) -> None:
        # only one dataset for train/test, no need to check stage.
        transform = Compose([Resize(tuple(self.data_conf.image_resize))])
        if self.data_conf.data_type == "sidd_small":
            self.dataset = SIDDSmall(
                self.data_conf.root_dir, self.data_conf.cache_file, transform=transform)
        else:
            raise Exception(
                f"data type {self.data_conf.data_type} is not implemented!")

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=1)
