"""
main.py

train the deep iamge prior model and get the denoised figure, calculate PSNR when required.
"""
import hydra
from pytorch_lightning import Trainer, seed_everything

from src.conf import Config
from src.data.datamodule import DeepImagePriorDataModule
from src.model.model import DeepImagePriorModel

import logging
logging.getLogger("lightning").setLevel(logging.ERROR)


@hydra.main(config_path=".", config_name="base_config", version_base="1.2")
def train_app(conf: Config) -> None:
    """ 
    The main train loop.
    """
    # * seed
    if conf.train.random_seed:
        seed_everything(conf.train.random_seed)

    for idx, fig_dir in enumerate((conf.data.root_dir/"Data").iterdir()):
        if fig_dir.is_dir():
            light_data = DeepImagePriorDataModule(conf.data, img_dir=fig_dir)
            light_data.setup(stage="fit")
            print(f"train fig{idx}, key: {light_data.dataset[0]['key']}")
            light_model = DeepImagePriorModel(conf)

            train_conf = conf.train
            trainer = Trainer(
                accelerator=train_conf.accelerator,
                devices=(
                    train_conf.distributed_devices if train_conf.accelerator == "gpu" else None),
                max_epochs=train_conf.epochs,
                num_sanity_val_steps=0,
                check_val_every_n_epoch=train_conf.check_val_every_n_epoch
            )
            trainer.fit(light_model, light_data)
        if idx == 1:
            break


if __name__ == "__main__":
    train_app()  # pylint: disable=no-value-for-parameter
