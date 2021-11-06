import numpy as np
import torch
nn = torch.nn
F = nn.functional

from sprelnet import util

def get_attn_net(net_HPs, dataset):
    kwargs = {
        "image_size": dataset["image size"],
        "num_labels": util.get_num_labels(dataset),
        "kernel_size": (net_HPs["relation kernel size"], net_HPs["relation kernel size"]),
        "num_heads": net_HPs["number of heads"],
        "num_relations": net_HPs["number of relations"],
        "miniseg_HPs": net_HPs["segmenter architecture"],
        "template_HPs": net_HPs["attnTemplate HPs"],
    }
    miniseg_HPs = kwargs["miniseg_HPs"]
    if isinstance(miniseg_HPs["channels by depth"], str):
        miniseg_HPs["channels by depth"] = miniseg_HPs["channels by depth"].replace("N_L", str(kwargs["num_labels"]))
        for k,v in miniseg_HPs.items():
            miniseg_HPs[k] = util.parse_int_list(v)

    template_HPs = kwargs["template_HPs"]
    if isinstance(template_HPs["channels by depth"], str):
        template_HPs["channels by depth"] = template_HPs["channels by depth"].replace("N_L*N_V", str(kwargs["num_labels"]*kwargs["num_heads"]))
        for k,v in template_HPs.items():
            template_HPs[k] = util.parse_int_list(v)

    return AttnNet(**kwargs).cuda()


def train_attnnet():
    return



class AttnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []
        self.layers = nn.Sequential(*self.layers)

    def forward(self, X):
        return self.layers(X)