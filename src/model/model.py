""" 
Implement the pytorch lightning model for the deep image prior. 
"""
from typing import Dict

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.conf import Config

MODELS = {
    "unet": smp.Unet,
    "unetpp": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "pan": smp.PAN,
    "deeplabv3": smp.DeepLabV3,
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
            self.models = nn.ModuleList([MODELS[self.model_conf.model_name](
                **MODEL_PARAMETERS[self.model_conf.model_name]) for _ in range(self.data_conf.dataset_size)])
        else:
            raise Exception(
                f"model {self.model_conf.model_name} is not implemented!")

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        noise, ref = batch["noise"], batch["input"]
        output = self.models[batch_idx](noise)
        loss = F.mse_loss(output, ref)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {
            "loss": loss
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_conf.lr)
        return {
            "optimizer": optimizer,
        }
