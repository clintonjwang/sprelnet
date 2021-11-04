import os, wandb
import numpy as np
import torch
import torchvision.datasets

from sprelnet import util

ds_folder = '/data/vision/polina/scratch/clintonw/datasets'
pixel_ds_path = os.path.join(ds_folder, 'pixel_ds.bin')
class_labels = {1:"base", 2:"downright", 3:"negative"}

def get_pixel_dataset(run=None, **kwargs):
    if os.path.exists(pixel_ds_path):
        if run is not None:
            artifact = wandb.Artifact('training_data', type='dataset')
            artifact.add_file(pixel_ds_path)
            run.log_artifact(artifact)
        return util.load_binary_file(pixel_ds_path)
    else:
        return create_pixel_dataset(**kwargs)


def create_pixel_dataset(N_train=9000, N_test=1000, N_labels=3, image_size=(16,16)):
    datapoints = []
    offsets = [(2,2), (-2,-1)]
    assert len(offsets) == N_labels-1
    for _ in range(N_train+N_test):
        dp = torch.zeros(N_labels, *image_size)
        for _ in range(5):
            i,j = np.random.randint(0, image_size[0]), np.random.randint(0, image_size[1])
            dp[0,i,j] = 1.
            if i+offsets[0][0] < image_size[0] and j+offsets[0][1] < image_size[1]:
                dp[1,i+offsets[0][0],j+offsets[0][1]] = 1.
            if i+offsets[1][0] < image_size[0] and j+offsets[1][1] < image_size[1]:
                dp[2,i+offsets[1][0],j+offsets[1][1]] = -1.
            if np.random.rand() > .5:
                break
        datapoints.append(dp)

    dataset = {"name": "pixels",
        "number of labels": 3,
        "image size": image_size, "train datapoints":datapoints[:N_train],
        "test datapoints":datapoints[N_train:],
        "offsets": offsets,
        "datapoint loader": lambda dp: (dp.sum(0, keepdim=True).cuda(), (dp != 0).float().cuda()),
    }
    util.save_binary_file(data=dataset, path=pixel_ds_path)
    return dataset
