import math
import numpy as np
import torch
nn = torch.nn
F = nn.functional

from sprelnet import util
from sprelnet.networks.relations import *
from sprelnet.networks.unet import *


def get_adv_sprelnet(net_HPs, dataset):
    # HPs = {**util.get_default_HPs(network_type), **HPs}
    kwargs = {
        "output_semantic": net_HPs["output semantic"],
        "image_size": dataset["image size"],
        "num_labels": util.get_num_labels(dataset),
        "kernel_size": (net_HPs["relation kernel size"], net_HPs["relation kernel size"]),
        "num_relations": net_HPs["number of relations"],
        "miniseg_HPs": net_HPs["segmenter architecture"],
        "type": net_HPs["type"],
    }
    miniseg_HPs = kwargs["miniseg_HPs"]
    if isinstance(miniseg_HPs["channels by depth"], str):
        miniseg_HPs["channels by depth"] = miniseg_HPs["channels by depth"].replace("N_L", str(kwargs["num_labels"]))
        for k,v in miniseg_HPs.items():
            miniseg_HPs[k] = util.parse_int_list(v)

    sprelnet = AdvSpRelNet(**kwargs).cuda()
    return sprelnet


class AdvSpRelNet(nn.Module):
    def __init__(self, image_size, num_labels, kernel_size=(9,9),
            num_relations=4, miniseg_HPs=None, multiscale=True, output_semantic=None, type=None):
        super().__init__()
        self.type = type
        self.N_r = num_relations #relations per label
        self.N_L = num_labels
        self.output_semantic = output_semantic
        self.kernel_size = kernel_size
        self.image_size = image_size

        if miniseg_HPs is None:
            self.G = MiniSeg()
        else:
            self.G = MiniSeg(miniseg_HPs["channels by depth"],
                    miniseg_HPs["kernels by depth"], miniseg_HPs["pool depths"])

        self.D_XY = match_XY(num_labels + 1)
        self.multiscale = multiscale

        if self.multiscale is True:
            if self.output_semantic == "Gaussian likelihood":
                self.relnet = rel_pyramid_likelihood(kernel_size, num_labels, N_levels=num_relations)
            elif self.output_semantic == "MAP estimate":
                self.relnet = rel_pyramid_estimator(kernel_size, num_labels, N_levels=num_relations)
            else:
                raise NotImplementedError
        else:
            if self.output_semantic == "MAP estimate":
                self.relnet = Id_Y(kernel_size, num_labels, num_relations)
            else:
                raise NotImplementedError

    def score(self, y):
        if self.multiscale is True:
            return self.relnet(y)
        else:
            y_down = F.avg_pool2d(y,2)
            return -((torch.sigmoid(self.relnet(y_down)) - y_down).abs().sum([1,2,3]))

    def forward(self, X):
        return self.G(X)


