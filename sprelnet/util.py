import os, glob, yaml, zipfile, shutil, wandb, subprocess
import dill as pickle
import numpy as np
import torch
nn = torch.nn
F = nn.functional

base_proj_dir = "/data/vision/polina/users/clintonw/code/sprelnet"
base_job_dir = "/data/vision/polina/users/clintonw/code/sprelnet/job_outputs"
entity_project = "spatial_relations/main_project"

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_num_labels(dataset):
    if dataset["name"] == "MNIST grid":
        return len(dataset["train label counts"])

    elif dataset["name"] == "pixels":
        return dataset["number of labels"]
        
    else:
        raise NotImplementedError

def end_slurm_jobs(jobs=None):
    if jobs is None:
        jobs = get_my_slurm_ids()
    else:
        jobs = [slurm_id for name,slurm_id in zip(get_my_slurm_jobs(), get_my_slurm_ids()) if name in jobs]

    for job in jobs:
        run_bash_cmd_with_stdout(f"scancel {job}")

def run_bash_cmd_with_stdout(command, grep=None):
    proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    out_lines = [line.decode("utf-8").rstrip() for line in proc.stdout.readlines()]
    if grep is None:
        return out_lines
    else:
        return [line for line in out_lines if grep in line]

def get_my_slurm_ids(ignore_name="jupyter"):
    lines = run_bash_cmd_with_stdout("squeue", grep="clintonw")
    lines = [line for line in lines if not ("PD" in line or ignore_name in line)]
    slurm_ids = [line.split()[0] for line in lines]
    return slurm_ids

def get_my_slurm_jobs(ignore_name="jupyter"):
    lines = run_bash_cmd_with_stdout("squeue", grep="clintonw")
    lines = [line for line in lines if not ("PD" in line or ignore_name in line)]
    slurm_ids = [line.split()[2] for line in lines]
    return slurm_ids

def rename_job(job, new_name):
    os.rename(f"{base_job_dir}/{job}", f"{base_job_dir}/{new_name}")


def get_code_directory():
    return f"{base_proj_dir}/sprelnet"

def get_config_path_for_job(job):
    return f"{base_job_dir}/{job}/config.yaml"

def get_hyperparameter_for_job(hyperparameter, job):
    config_path = get_config_path_for_job(job)
    try:
        with open(config_path, 'r') as stream:
            args = yaml.safe_load(stream)
    except FileNotFoundError:
        print(f"could not find {config_path}")
        return
    return get_nested_attr(args, hyperparameter)

def get_run_id_for_job(job):
    return get_hyperparameter_for_job("run_id", job)

def get_run_path_for_job(job):
    return get_hyperparameter_for_job("run_path", job)

def iou(pred_seg, gt_seg):
    with torch.no_grad():
        return ((pred_seg & gt_seg).sum(axis=(1,2)) / (pred_seg | gt_seg).sum(axis=(1,2)))

def dice(pred_seg, gt_seg):
    with torch.no_grad():
        return (2*(pred_seg & gt_seg).sum(axis=(1,2)) / (
            pred_seg.sum(axis=(1,2)) + gt_seg.sum(axis=(1,2))))

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


def delete_job_outputs(job_name=None):
    api = wandb.Api()
    if job_name is None:
        return delete_job_outputs("*")
    elif "*" in job_name:
        for path in glob.glob(f"{base_job_dir}/{job_name}"):
            print(f"deleting {path}")
            delete_job_outputs(os.path.basename(path))
    else:
        path = f"{base_job_dir}/{job_name}"
        
        run_path = get_run_path_for_job(job_name)
        if run_path is None:
            return
        try:
            run = api.run(run_path)
            run.delete(delete_artifacts=True)
        except:
            pass

        shutil.rmtree(path)


def parse_int_list(x):
    # converts string to a list of ints
    if not isinstance(x, str):
        return [x]
    try:
        return [int(x)]
    except ValueError:
        return [int(s.strip()) for s in x.split(',')]
# def parse_int_or_list(x):
#     # converts string to an int or list of ints
#     if not isinstance(x, str):
#         return x
#     try:
#         return int(x)
#     except ValueError:
#         return [int(s.strip()) for s in x.split(',')]


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


class BasicDS(torch.utils.data.Dataset):
    def __init__(self, datapoints, dp_loader):
        self.datapoints = datapoints
        self.dp_loader = dp_loader
    def __getitem__(self, index):
        return self.dp_loader(self.datapoints[index])
    def __len__(self):
        return len(self.datapoints)
