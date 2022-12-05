""" 
Util functions for visualizing the figures.
"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T


def show(imgs: List[torch.Tensor]):
    """ 
    Show a list of tensor images
    """
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.show()
