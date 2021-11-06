import argparse, itertools, os, shutil, yaml, math, wandb
import numpy as np
import torch
nn = torch.nn
F = nn.functional
import bitsandbytes as bnb

torch.backends.cudnn.deterministic = True

from sprelnet import util, losses, wab, visualize
from sprelnet.data import mnist, pixels
from sprelnet.networks.relations import get_relnet
from sprelnet.networks.unet import get_unet
from sprelnet.networks.adversarial import get_adv_sprelnet
from sprelnet.networks.contrastive import get_contra_net
from sprelnet.networks.patch_net import get_patch_net


def train_patchnet(paths, loss_settings, optimizer_settings, network, dataloaders, dataset):
    batch_size = dataloaders["train"].batch_size
    N_train = len(dataset["train datapoints"])
    N_test = len(dataset["test datapoints"])
    loss_weights = loss_settings["weights"]

    optimizer = bnb.optim.Adam8bit(network.parameters(), lr=float(optimizer_settings["learning rate"]))
    K = [k//2 for k in network.rel_kernel_size]

    down4 = nn.AvgPool2d((4,4))
    down2 = nn.AvgPool2d((2,2))

    for epoch in range(1, optimizer_settings["max epochs"]+1):
        network["training event"]["epoch"] = epoch
        loss1_sum, loss2_sum = 0,0
        n_batches = math.ceil(N_train/batch_size)

        for batch in train_dataloader:
            X,Y_gt = batch
            Y_0 = network.init_guess(X)
            if epoch <= optimizer_settings["phase 1 end"]:
                Y_hat = network(X, Y_gt)
                Y_hat = down4(Y_hat)
                Y_0 = down4(Y_0)
                Y_gt = down4(Y_gt)
                # loss2 = bce_loss(Y_0, Y_gt)

            elif epoch < optimizer_settings["phase 2 end"]:
                # goes from 1 -> 0
                a = (optimizer_settings["phase 2 end"]-epoch) / (optimizer_settings["phase 2 end"]-optimizer_settings["phase 1 end"])
                Y_mix = a*Y_gt + (1-a)*Y_0
                Y_hat = network(X, Y_mix)
                for _ in range(optimizer_settings["number of refinements"]):
                    Y_hat = network(X, Y_hat)
                Y_hat = down2(Y_hat)
                Y_0 = down2(Y_0)
                Y_gt = down2(Y_gt)
                # loss2 = bce_loss(Y_0, Y_gt) * a

            else:
                Y_hat = network(X, Y_0)
                for _ in range(optimizer_settings["number of refinements"]):
                    Y_hat = network(X, Y_hat)
                # loss2 = torch.zeros(1).float().cuda()

            loss1 = bce_loss(Y_hat, Y_gt)
            loss2 = bce_loss(Y_0, Y_gt)
            loss1_sum += loss1.item()
            loss2_sum += loss2.item()
            loss = loss1 + loss2 * loss_weights["init guess weight"]

            loss.backward()
            sparse_reg, smooth_reg = losses.get_single_scale_kernel_regs(relnet, loss_weights)
            sparse_reg.backward()
            smooth_reg.backward()
            for i in range(network.N_L):
                network.r_m.weight.grad[i:W.size(0):network.N_L, i, K[0],K[1]] = 0 # "mask" the inpainted patch

            optimizer.step()

        wandb.log({
            "final seg loss": loss1_sum/n_batches,
            "init guess loss": loss2_sum/n_batches,
            "sparse kernel regularization": sparse_reg.item(),
            "smooth kernel regularization": smooth_reg.item(),
        })

        if epoch % optimizer_settings["validation frequency"] == 0:
            network.eval()
            loss_sum = 0
            n_batches = math.ceil(N_test/batch_size)
            for batch in test_dataloader:
                X,Y_gt = batch
                Y_hat = network(X)
                for _ in range(optimizer_settings["number of refinements"]):
                    Y_hat = network(X, Y_hat)
                loss = bce_loss(Y_hat, Y_gt)
                loss_sum += loss.item()
            test_loss = loss_sum/n_batches

            network.train()

        if epoch % optimizer_settings["checkpoint frequency"] == 0:
            wab.save_state(network, paths, run, model_artifact)
    wab.save_state(network, paths, run, model_artifact)

def train_contranet(paths, loss_settings, optimizer_settings, network, dataloaders, dataset):
    batch_size = dataloaders["train"].batch_size
    N_train = len(dataset["train datapoints"])
    N_test = len(dataset["test datapoints"])
    loss_weights = loss_settings["weights"]

    for epoch in range(1, optimizer_settings["max epochs"]+1):
        network["training event"]["epoch"] = epoch
        loss1_sum, loss2_sum = 0,0
        n_batches = math.ceil(N_train/batch_size)

        for batch in train_dataloader:
            pass

        if epoch % optimizer_settings["checkpoint frequency"] == 0:
            wab.save_state(network, paths, run, model_artifact)
    wab.save_state(network, paths, run, model_artifact)

def train_main_relnet(paths, loss_settings, optimizer_settings, network, dataloaders, dataset):
    # learns spatial relations and segmentation task simultaneously
    
    G = network.G
    relnet = network.relnet
    if network.type == "adversarial":
        D = network.D_XY
    loss_weights = loss_settings["weights"]

    # setup dataloaders
    batch_size = dataloaders["train"].batch_size
    N_train = len(dataset["train datapoints"])
    N_test = len(dataset["test datapoints"])
    bce_loss = losses.get_bce_loss(dataset)

    G_optim = bnb.optim.Adam8bit(G.parameters(), lr=float(optimizer_settings["G learning rate"]))
    if network.type == "adversarial":
        DR_optim = bnb.optim.Adam8bit(itertools.chain(relnet.parameters(), D.parameters()),
            lr=float(optimizer_settings["D learning rate"]))
    else:
        DR_optim = bnb.optim.Adam8bit(relnet.parameters(), lr=float(optimizer_settings["relnet learning rate"]))

    loss_names = ["rel true", "rel fake", "seg loss",
        "sparse kernel regularization", "smooth kernel regularization"]
    if network.type == "adversarial":
        loss_names += ["-log pxy true", "-log pxy fake"]

    model_artifact = wandb.Artifact("sprelnet", type="model", description="segmenter with spatial relations")
    model_artifact.add_dir(paths["model weights directory"])

    for epoch in range(1, optimizer_settings["epochs"]+1):
        pxy_true_sum, pxy_fake_sum, rel_true_sum, rel_fake_sum, seg_loss_sum = 0,0, 0,0, 0
        n_batches = math.ceil(N_train/batch_size)
        rel_weight = losses.determine_loss_weight("relation score", epoch=epoch, loss_settings=loss_settings)

        for batch in dataloaders["train"]:
            X,Y_gt = batch
            Y_logits = G(X)
            seg_loss = bce_loss(Y_logits, Y_gt)

            Y_hat = torch.sigmoid(Y_logits)
            if network.type == "adversarial":
                p_xy_fake = D(X, Y_hat).mean()
                pxy_fake_sum += p_xy_fake.item()

            rel_fake = relnet(Y_hat).mean()
            rel_fake_sum += rel_fake.item()
            seg_loss_sum += seg_loss.item()

            if network.type == "adversarial":
                G_loss = seg_loss * loss_weights["cross entropy"] + rel_fake * rel_weight - p_xy_fake
            else:
                G_loss = seg_loss * loss_weights["cross entropy"] + rel_fake * rel_weight

            G_optim.zero_grad()
            G_loss.backward(retain_graph=True)
            G_optim.step()




            if network.type == "adversarial":
                p_xy_true = D(X, Y_gt).mean()
                p_xy_fake = D(X, Y_hat.detach()).mean()
                pxy_true_sum += p_xy_true.item()
            rel_true = relnet(Y_gt).mean() * loss_weights["relation score"]
            rel_true_sum += rel_true.item()

            if network.type == "adversarial":
                DR_loss = p_xy_fake - p_xy_true + rel_true
            else:
                DR_loss = rel_true

            DR_optim.zero_grad()
            DR_loss.backward()
            sparse_reg, smooth_reg = losses.get_multiscale_kernel_regs(relnet, loss_weights)
            sparse_reg.backward()
            smooth_reg.backward()
            util.mask_identity_grad_in_kernel(relnet)
            DR_optim.step()


        train_loss = {
            "rel true": rel_true_sum/n_batches,
            "rel fake": rel_fake_sum/n_batches,
            "seg loss": seg_loss_sum/n_batches,
            "sparse kernel regularization": sparse_reg.item(),
            "smooth kernel regularization": smooth_reg.item(),
        }
        if network.type == "adversarial":
            train_loss["-log pxy true"] = -pxy_true_sum/n_batches
            train_loss["-log pxy fake"] = -pxy_fake_sum/n_batches
        wandb.log(train_loss)

        if epoch % optimizer_settings["validation frequency"] == 0:
            wab.log_sample_outputs(X, Y_gt, Y_hat, dataset=dataset, phase="training", n=2)
            
            network.eval()

            wab.log_relnet(relnet, dataset)
            loss_sum = 0
            n_batches = math.ceil(N_test/batch_size)
            dices = []
            for batch in dataloaders["test"]:
                X,Y_gt = batch
                Y_logits = G(X)
                loss = bce_loss(Y_logits, Y_gt)
                Y_hat = torch.sigmoid(Y_logits)
                loss_sum += loss.item()
                dices.append(losses.dice(Y_hat > .5, Y_gt > .5))

            wab.update_metric("loss", loss_sum/n_batches)
            wab.update_metric("dice", torch.cat(dices, dim=0).mean())
            wab.log_sample_outputs(X, Y_gt, Y_hat, dataset=dataset, phase="validation", n=2)
            network.train()

        if epoch % optimizer_settings["checkpoint frequency"] == 0:
            wab.save_state(network, paths, run, model_artifact)
    wab.save_state(network, paths, run, model_artifact)


def train_relnet(paths, loss_settings, optimizer_settings, network, dataloaders, dataset):
    # spatial relations only, no segmentations
    
    batch_size = dataloaders["train"].batch_size
    N_train = len(dataset["train datapoints"])
    N_test = len(dataset["test datapoints"])
    loss_weights = loss_settings["weights"]

    optimizer = bnb.optim.Adam8bit(network.parameters(), lr=float(optimizer_settings["learning rate"]))

    relnet = network
    model_artifact = wandb.Artifact("relnet", type="model", description="segmenter with spatial relations")
    model_artifact.add_dir(paths["model weights directory"])

    for epoch in range(1, optimizer_settings["epochs"]+1):
        loss_sum = 0
        n_batches = math.ceil(N_train/batch_size)

        for batch in dataloaders["train"]:
            optimizer.zero_grad()
            if not ("debug" in loss_settings and loss_settings["debug"] == "regs only"):
                _,Y_gt = batch
                loss = network(Y_gt).mean()
                loss_sum += loss.item()
                loss.backward()

            sparse_reg, smooth_reg = losses.get_multiscale_kernel_regs(network, loss_weights)
            sparse_reg.backward()
            smooth_reg.backward()
            util.mask_identity_grad_in_kernel(network)
            optimizer.step()

        wandb.log({
            "relation loss": loss_sum/n_batches,
            "sparse reg": sparse_reg.item(),
            "smooth reg": smooth_reg.item(),
        })

        if epoch % optimizer_settings["validation frequency"] == 0:
            network.eval()
            wab.log_relnet(network, dataset)
            network.train()

        if epoch % optimizer_settings["checkpoint frequency"] == 0:
            wab.save_state(network, paths, run, model_artifact)
    
    wab.save_state(network, paths, run, model_artifact)



def train_unet(paths, loss_settings, optimizer_settings, network, dataloaders, dataset):
    # standard segmentation U-Net with no spatial relations

    batch_size = dataloaders["train"].batch_size
    N_train = len(dataset["train datapoints"])
    N_test = len(dataset["test datapoints"])
    bce_loss = losses.get_bce_loss(dataset)

    optimizer = bnb.optim.Adam8bit(network.parameters(), lr=float(optimizer_settings["learning rate"]))

    model_artifact = wandb.Artifact("baseline", type="model", description="segmenter with spatial relations")
    model_artifact.add_dir(paths["model weights directory"])

    for epoch in range(1, optimizer_settings["epochs"]+1):
        loss_sum = 0
        n_batches = math.ceil(N_train/batch_size)

        for batch in dataloaders["train"]:
            optimizer.zero_grad()
            X,Y_gt = batch
            Y_logits = network(X)
            loss = bce_loss(Y_logits, Y_gt)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

        wandb.log({"bce": loss_sum/n_batches})

        if epoch % optimizer_settings["validation frequency"] == 0:
            network.eval()
            Y_hat = torch.sigmoid(Y_logits)
            wab.log_sample_outputs(X, Y_gt, Y_hat, dataset=dataset, phase="training", n=2)

            loss_sum = 0
            n_batches = math.ceil(N_test/batch_size)

            dices = []
            for batch in dataloaders["test"]:
                X,Y_gt = batch
                Y_logits = network(X)
                loss = bce_loss(Y_logits, Y_gt)
                loss_sum += loss.item()
                Y_hat = torch.sigmoid(Y_logits)

                dices.append(losses.dice(Y_hat > .5, Y_gt > .5))

            wab.update_metric("dice", torch.cat(dices, dim=0).mean())
            wab.update_metric("loss", loss_sum/n_batches)

            wab.log_sample_outputs(X, Y_gt, Y_hat, dataset=dataset, phase="validation", n=2)
            network.train()

        if epoch % optimizer_settings["checkpoint frequency"] == 0:
            wab.save_state(network, paths, run, model_artifact)
    wab.save_state(network, paths, run, model_artifact)



def prep_run(cmd_args, args):    
    if "tags" not in args:
        args["tags"] = []

    config = {k:args[k] for k in ["network", "loss", "optimizer", "data loading"]}
    run = wandb.init(project="main_project", config=config, tags=args["tags"],
            name=cmd_args.job_id, entity="spatial_relations",
            settings=wandb.Settings(start_method="fork"))

    # fill in paths
    paths = args["paths"]
    paths["job output directory"] = util.get_job_output_folder(cmd_args.job_id)
    # paths["loss history path"] = os.path.join(paths["job output directory"], "losses.bin")
    paths["model weights directory"] = os.path.join(paths["job output directory"], "models")
    if not os.path.exists(paths["model weights directory"]):
        os.makedirs(paths["model weights directory"])

    # copy code base and config file to slurm output
    # util.save_code_dir(util.get_code_directory(), paths["job output directory"])
    config_target_path = os.path.join(paths["job output directory"], "config.yaml")
    shutil.copy(cmd_args.config_path, config_target_path)
    with open(config_target_path, "a") as f:
        f.write(f"\nrun_id: {os.path.basename(run.path)}")
        f.write(f"\nrun_path: {run.path}")

    util.set_random_seed(args["random seed"])
    if args["fixed input size"] is True:
        torch.backends.cudnn.benchmark = True

    return run, args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path')
    parser.add_argument('--job_id', default='test_00')
    cmd_args = parser.parse_args()
    # class cmd_args:
    #     job_id = "test_00"
    #     config_path = f"{util.base_proj_dir}/configs/{job_id}.yaml"
    #     os.makedirs(f"{util.base_job_dir}/{job_id}")
    with open(cmd_args.config_path, 'r') as stream:
        args = yaml.safe_load(stream)
        
    run, args = prep_run(cmd_args, args)

    if args["data loading"]["dataset"] == "MNIST grid":
        dataset = mnist.get_multi_mnist(run=run)
    elif args["data loading"]["dataset"] == "pixels":
        dataset = pixels.get_pixel_dataset(run=run)
    else:
        raise NotImplementedError("bad dataset")

    dataloaders = util.get_dataloaders(dataset, batch_size=args["data loading"]["batch size"])

    if args["network"]["type"] in ["adversarial", "vanilla"]:
        get_net_fxn = get_adv_sprelnet
        train_fxn = train_main_relnet
    elif args["network"]["type"] == "unet":
        get_net_fxn = get_unet
        train_fxn = train_unet
    elif args["network"]["type"] == "relnet only":
        get_net_fxn = get_relnet
        train_fxn = train_relnet
    elif args["network"]["type"] == "contrastive":
        get_net_fxn = get_contra_net
        train_fxn = train_contranet
    elif args["network"]["type"] == "iterative with attention over patch proposals":
        get_net_fxn = get_patch_net
        train_fxn = train_patchnet
    else:
        raise NotImplementedError("bad network type")


    network = get_net_fxn(net_HPs=args["network"], dataset=dataset)
    train_fxn(paths=args["paths"],
        loss_settings=args["loss"], optimizer_settings=args["optimizer"],
        dataloaders=dataloaders, network=network, dataset=dataset)
