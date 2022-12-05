"""
main.py

train the deep iamge prior model and get the denoised figure, calculate PSNR when required.
"""
import hydra
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary

from src.conf import Config
from src.data.datamodule import DeepImagePriorDataModule
from src.model.model import DeepImagePriorModel


@hydra.main(config_path=".", config_name="base_config", version_base="1.2")
def train_app(conf: Config) -> None:
    """ 
    The main train loop.
    """
    # * seed
    if conf.train.random_seed:
        seed_everything(conf.train.random_seed)

    # * prepare light data
    light_data = DeepImagePriorDataModule(conf.data)
    light_data.prepare_data()

    # * prepare model
    light_model = DeepImagePriorModel(conf)

    # * callbacks
    callbacks = []
    callbacks.append(LearningRateMonitor(
        logging_interval='epoch'))
    callbacks.append(ModelSummary(max_depth=2))

    # * trainer
    train_conf = conf.train
    trainer = Trainer(
        callbacks=callbacks,
        accelerator=train_conf.accelerator,
        devices=(
            train_conf.distributed_devices if train_conf.accelerator == "gpu" else None),
        max_epochs=train_conf.epochs,
        strategy=(train_conf.strategy if train_conf.accelerator ==
                  "gpu" else None),
        num_sanity_val_steps=0,  # no need to do this check outside development
    )

    # * train
    light_data.setup(stage="fit")
    trainer.fit(light_model, light_data)


if __name__ == "__main__":
    train_app()  # pylint: disable=no-value-for-parameter
