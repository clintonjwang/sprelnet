import shutil
import numpy as np
import torch
nn = torch.nn
F = nn.functional

def train_model(paths, aug_settings, loss_settings, optimizer_settings, network_settings, dataloaders):
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train UNet for placenta segmentation on MRI')
    parser.add_argument('--config_path', dest='config_path')
    parser.add_argument('--job_id', dest='job_id', default='test_00')
    cmd_args = parser.parse_args()
    # class cmd_args:
    #     config_path = "/data/vision/polina/users/clintonw/code/placenta/train_placenta_config.yaml"
    #     job_id = "test_00"
    with open(cmd_args.config_path, 'r') as stream:
        args = yaml.safe_load(stream)

    # fill in paths
    paths = args["paths"]
    if paths["path to initial model weights"].lower() == "none":
        paths["path to initial model weights"] = None
    paths["job output directory"] = os.path.join(paths["slurm output directory"], cmd_args.job_id)
    paths["loss history path"] = os.path.join(paths["job output directory"], "output.csv")
    paths["model weights directory"] = os.path.join(paths["job output directory"], "models")
    if not os.path.exists(paths["model weights directory"]):
        os.makedirs(paths["model weights directory"])
    args["optimizer"]["learning rate"] = float(args["optimizer"]["learning rate"])

    # copy code base and config file to slurm output
    util.save_code_dir(paths["code directory"], paths["job output directory"])
    shutil.copy(cmd_args.config_path, os.path.join(paths["job output directory"], "config.yaml"))
    
    # set random seed
    np.random.seed(args["random seed"])
    torch.manual_seed(args["random seed"])

    dataloaders = util.get_dataloaders()

    print('Starting training')
    train_model(paths=args["paths"], aug_settings=args["augmentations"],
        loss_settings=args["loss"], optimizer_settings=args["optimizer"],
        network_settings=args["network"], dataloaders=dataloaders)
