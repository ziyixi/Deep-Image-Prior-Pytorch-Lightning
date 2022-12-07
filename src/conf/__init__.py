""" 
configureation for the entire project.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from hydra.conf import HydraConf, JobConf, RunDir
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ModelConfig:
    """
    Specify the model and the architecture used in the project.
    """
    model_name: str = "unet"


@dataclass
class TrainConfig:
    """ 
    Specify the training details.
    """
    epochs: int = 100
    random_seed: Optional[int] = 1234
    lr: float = 2e-4
    accelerator: str = "gpu"
    distributed_devices: List[int] = field(
        default_factory=lambda: [0])
    check_val_every_n_epoch: int = 50


@dataclass
class DataConfig:
    """ 
    Specify the data details.
    """
    batch_size: int = 1
    data_type: str = "sidd_small"
    root_dir: Path = Path("")
    image_resize: List[int] = field(default_factory=lambda: [1600, 2080])
    dataset_size: int = 160
    dataset_channel: int = 3
    mask_threshold: float = 0.5

    fig_save_dir: Path = Path("")
    total_run_figs_number: int = 20


# * ======================================== * #
# * main conf
defaults = [
    {"data": "base_data"},
    {"model": "base_model"},
    {"train": "base_train"},
    "_self_"
]


@dataclass
class Hydra(HydraConf):
    """ 
    Customed project setting    
    """
    # run: RunDir = RunDir("${output_dir}")
    run: RunDir = RunDir(
        dir="outputs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}")
    job: JobConf = JobConf(chdir=True)


@dataclass
class Config:
    """ 
    The main config for the project.
    """
    defaults: List[Any] = field(default_factory=lambda: defaults)
    # * custom
    hydra: Hydra = Hydra()
    # * settings
    data: DataConfig = MISSING
    model: ModelConfig = MISSING
    train: TrainConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="data", name="base_data", node=DataConfig)
cs.store(group="model", name="base_model", node=ModelConfig)
cs.store(group="train", name="base_train", node=TrainConfig)
