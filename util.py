import yaml
import numpy as np
import torch
nn = torch.nn
F = nn.functional


def get_default_HPs(network_type):
    path = "/data/vision/polina/users/clintonw/code/sprelnet/default_HPs.yaml"
    with open(path, 'r') as stream:
        data = yaml.safe_load(stream)
    if "SpRelNet" in network_type:
        return {**data["spatial relation net"], **data[network_type]}
    else:
        return data[network_type]

def get_dataloaders():
    return

def save_code_dir():
    return
