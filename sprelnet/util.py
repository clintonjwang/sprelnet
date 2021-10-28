import os, glob, yaml, zipfile, shutil
import dill as pickle
import numpy as np
import torch
nn = torch.nn
F = nn.functional

base_job_dir = "/data/vision/polina/users/clintonw/code/sprelnet/job_outputs"
# def parse_int_or_list(x):
#     # converts string to an int or list of ints
#     if not isinstance(x, str):
#         return x
#     try:
#         return int(x)
#     except ValueError:
#         return [int(s.strip()) for s in x.split(',')]

def iou(pred_seg, gt_seg):
    return np.nan_to_num((pred_seg & gt_seg).sum(axis=(1,2)) / (pred_seg | gt_seg).sum(axis=(1,2)), 0, 0, 0)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def get_nested_attr(x, keys):
    if isinstance(keys, str):
        return x[keys]
    ret = x
    for k in keys:
        ret = ret[k]
    return ret

def print_job_log(job_name):
    path = f"{base_job_dir}/{job_name}/logs.err"
    with open(path, "r") as f:
        return f.readlines()

def delete_job_outputs(job_name):
    if "*" in job_name:
        for path in glob.glob(f"{base_job_dir}/{job_name}"):
            print(f"deleting {path}")
            shutil.rmtree(path)
    else:
        path = f"{base_job_dir}/{job_name}"
        shutil.rmtree(path)

def parse_int_list(x):
    # converts string to a list of ints
    if not isinstance(x, str):
        return [x]
    try:
        return [int(x)]
    except ValueError:
        return [int(s.strip()) for s in x.split(',')]

def save_code_dir(code_dir, save_dir):
    root_path = os.path.dirname(code_dir)
    zipf = zipfile.ZipFile(os.path.join(save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(code_dir):
        # if root.endswith("job_outputs"):
        #     continue
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.join('code/',os.path.relpath(file_path, root_path))
            zipf.write(file_path, arcname=relative_path) 
    zipf.close()

def get_job_output_folder(job):
    return f"{base_job_dir}/{job}"

def get_losses_for_spatial_relation_net_job(job):
    return load_binary_file(os.path.join(get_job_output_folder(job), "losses.bin"))


# def get_default_HPs(network_type):
#     path = "/data/vision/polina/users/clintonw/code/sprelnet/default_HPs.yaml"
#     with open(path, 'r') as stream:
#         data = yaml.safe_load(stream)
#     if "SpRelNet" in network_type:
#         return {**data["spatial relation net"], **data[network_type]}
#     else:
#         return data[network_type]

def save_binary_file(path, data):
    with open(path, "wb") as opened_file:
        pickle.dump(data, opened_file)

def load_binary_file(path):
    with open(path, "rb") as opened_file:
        return pickle.load(opened_file)

def get_dataloaders(dataset, batch_size):
    train_dataloader = create_dataloader_for_datapoints(
        dataset["train datapoints"], dp_loader=dataset["datapoint loader"], batch_size=batch_size)
    test_dataloader = create_dataloader_for_datapoints(
        dataset["test datapoints"], dp_loader=dataset["datapoint loader"], batch_size=batch_size)
    return {"train": train_dataloader, "test": test_dataloader}

def create_dataloader_for_datapoints(datapoints, dp_loader, **kwargs):
    torch_ds = BasicDS(datapoints, dp_loader)
    return torch.utils.data.DataLoader(torch_ds, **kwargs)

def sample_without_replacement(collection):
    raise NotImplementedError


class BasicDS(torch.utils.data.Dataset):
    def __init__(self, datapoints, dp_loader):
        self.datapoints = datapoints
        self.dp_loader = dp_loader
    def __getitem__(self, index):
        return self.dp_loader(self.datapoints[index])
    def __len__(self):
        return len(self.datapoints)
