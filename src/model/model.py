""" 
Implement the pytorch lightning model for the deep image prior. 
"""
from typing import Dict

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio
from torchvision.utils import save_image

from src.conf import Config

MODELS = {
    "unet": smp.Unet,
    "unetpp": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "deeplabv3p": smp.DeepLabV3Plus
}

MODEL_PARAMETERS = {
    "unet": {
        "encoder_weights": None,
        "classes": 3,
        "activation": "sigmoid"
    },
    "unetpp": {
        "encoder_weights": None,
        "classes": 3,
        "activation": "sigmoid"
    },
    "manet": {
        "encoder_weights": None,
        "classes": 3,
        "activation": "sigmoid"
    },
    "linknet": {
        "encoder_weights": None,
        "classes": 3,
        "activation": "sigmoid"
    },
    "fpn": {
        "encoder_weights": None,
        "classes": 3,
        "activation": "sigmoid"
    },
    "pspnet": {
        "encoder_weights": None,
        "classes": 3,
        "activation": "sigmoid"
    },
    "pan": {
        "encoder_weights": None,
        "classes": 3,
        "activation": "sigmoid"
    },
    "deeplabv3": {
        "encoder_weights": None,
        "classes": 3,
        "activation": "sigmoid"
    },
    "deeplabv3p": {
        "encoder_weights": None,
        "classes": 3,
        "activation": "sigmoid"
    },
}


class DeepImagePriorModel(pl.LightningModule):
    def __init__(self, conf: Config) -> None:
        super().__init__()
        # * create the image to image model
        # * the model will repeat dataset_size times for each batch (batch_size == 1)
        self.model_conf = conf.model
        self.data_conf = conf.data
        self.train_conf = conf.train

        if self.model_conf.model_name in MODELS:
            self.image_model = MODELS[self.model_conf.model_name](
                **MODEL_PARAMETERS[self.model_conf.model_name])
        else:
            raise Exception(
                f"model {self.model_conf.model_name} is not implemented!")

    def training_step(self, batch: Dict, batch_idx: int):
        noise_input, noise_fig = batch["noise_input"], batch["noise_fig"]
        output = self.image_model(noise_input)
        loss = F.mse_loss(output, noise_fig)
        return {
            "loss": loss
        }

    def validation_step(self, batch: Dict, batch_idx: int):
        noise_input, noise_fig, target_fig, key = batch["noise_input"], batch[
            "noise_fig"], batch["target_fig"], batch["key"]
        output = self.image_model(noise_input)
        psnr = peak_signal_noise_ratio(output, target_fig)
        # * save fig
        fig_name = self.data_conf.fig_save_dir / \
            f"{key[0]}_step{self.current_epoch}_psnr{psnr:.2f}.png"
        save_image(output[0], fig_name)
        # * save raw and target
        raw_psnr = peak_signal_noise_ratio(noise_fig, target_fig)
        raw_fig_name = self.data_conf.fig_save_dir / \
            f"{key[0]}_step0_psnr{raw_psnr:.2f}.png"
        if not raw_fig_name.exists():
            save_image(noise_fig[0], raw_fig_name)

        target_psnr = peak_signal_noise_ratio(target_fig, target_fig)
        target_fig_name = self.data_conf.fig_save_dir / \
            f"{key[0]}_stepinf_psnr{target_psnr:.2f}.png"
        if not target_fig_name.exists():
            save_image(target_fig[0], target_fig_name)

        return {
            "psnr": psnr
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_conf.lr)
        return {
            "optimizer": optimizer,
        }
