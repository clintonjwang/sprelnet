import argparse, itertools, os, shutil, yaml, math, wandb
import numpy as np
import torch
nn = torch.nn
F = nn.functional

torch.backends.cudnn.deterministic = True

from sprelnet import util, losses, wab, visualize
from sprelnet.data import mnist, pixels
from sprelnet.networks.relations import get_relnet
from sprelnet.networks.unet import get_unet
from sprelnet.networks.adversarial import get_adv_sprelnet
from sprelnet.networks.contrastive import get_contra_net, train_contranet
from sprelnet.networks.patch_net import get_patch_net, train_patchnet


def train_main_relnet(paths, loss_settings, optimizer_settings, network, dataloaders, dataset):
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

    G_optim = torch.optim.Adam(G.parameters(), lr=float(optimizer_settings["G learning rate"]))
    if network.type == "adversarial":
        DR_optim = torch.optim.Adam(itertools.chain(relnet.parameters(), D.parameters()),
            lr=float(optimizer_settings["D learning rate"]))
    else:
        DR_optim = torch.optim.Adam(relnet.parameters(), lr=float(optimizer_settings["relnet learning rate"]))

    loss_names = ["rel true", "rel fake", "seg loss",
        "sparse kernel regularization", "smooth kernel regularization"]
    if network.type == "adversarial":
        loss_names += ["-log pxy true", "-log pxy fake"]

    train_losses = {
        k:[] for k in loss_names
    }
    test_losses = []

    # static regularizers
    K = [k//2 for k in relnet.kernel_size]
    for epoch in range(1, optimizer_settings["epochs"]+1):
        pxy_true_sum, pxy_fake_sum, rel_true_sum, rel_fake_sum, seg_loss_sum = 0,0, 0,0, 0
        n_batches = math.ceil(N_train/batch_size)

        for batch in dataloaders["train"]:
            X,Y_gt = batch
            Y_logits = G(X)
            Y_hat = torch.sigmoid(Y_logits)

            if network.type == "adversarial":
                p_xy_fake = D(X, Y_hat).mean()
                pxy_fake_sum += p_xy_fake.item()

            rel_fake = relnet(Y_hat).mean() * loss_weights["relation score"]
            seg_loss = bce_loss(Y_logits, Y_gt) * loss_weights["cross entropy"]

            G_optim.zero_grad()
            if network.type == "adversarial":
                G_loss = seg_loss + rel_fake - p_xy_fake
            else:
                G_loss = seg_loss + rel_fake
            G_loss.backward(retain_graph=True)
            G_optim.step()

            rel_fake_sum += rel_fake.item()
            seg_loss_sum += seg_loss.item()


            if network.type == "adversarial":
                p_xy_true = D(X, Y_gt).mean()
                p_xy_fake = D(X, Y_hat.detach()).mean()
                pxy_true_sum += p_xy_true.item()

            rel_true = relnet(Y_gt).mean() * loss_weights["relation score"]
            rel_true_sum += rel_true.item()

            DR_optim.zero_grad()
            if network.type == "adversarial":
                DR_loss = p_xy_fake - p_xy_true + rel_true
            else:
                DR_loss = rel_true
            DR_loss.backward()
            sparse_reg, smooth_reg = losses.get_multiscale_kernel_regs(relnet, loss_weights)
            sparse_reg.backward()
            smooth_reg.backward()
            for i in range(network.N_L):
                for r_m in relnet.pyramid:
                    r_m.weight.grad[i, i, K[0],K[1]] = 0 # "mask" the inpainted patch
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
        for k,v in train_loss.items():
            train_losses[k].append(v)

        if epoch % optimizer_settings["validation frequency"] == 0:
            wab.log_sample_outputs(X[0], Y_gt[0], Y_hat[0], dataset=dataset, name="training 1")
            wab.log_sample_outputs(X[1], Y_gt[1], Y_hat[1], dataset=dataset, name="training 2")
            
            network.eval()

            wab.log_relnet(relnet, dataset)
            loss_sum = 0
            n_batches = math.ceil(N_test/batch_size)
            for batch in dataloaders["test"]:
                X,Y_gt = batch
                Y_logits = G(X)
                loss = bce_loss(Y_logits, Y_gt)
                loss_sum += loss.item()
            test_loss = loss_sum/n_batches

            wandb.log({"test loss": test_loss})
            wab.log_sample_outputs(X[0], Y_gt[0], torch.sigmoid(Y_logits[0]), dataset=dataset, name="validation 1")
            wab.log_sample_outputs(X[1], Y_gt[1], torch.sigmoid(Y_logits[1]), dataset=dataset, name="validation 2")

            network.train()

        # if epoch % optimizer_settings["checkpoint frequency"] == 0:
        #     torch.save(network.state_dict(),
        #         os.path.join(paths["model weights directory"], f"{epoch}.state"))
        #     util.save_binary_file(data=network,
        #         path=os.path.join(paths["model weights directory"], f"{epoch}.inst"))

    # loss_dict = {"train losses":train_losses, "test losses":test_losses}
    # torch.save(network.state_dict(), os.path.join(paths["model weights directory"], f"{epoch}.state"))
    # util.save_binary_file(data=network, path=os.path.join(paths["model weights directory"], f"{epoch}.inst"))
    # return results


def train_relnet(paths, loss_settings, optimizer_settings, network, dataloaders, dataset):
    batch_size = dataloaders["train"].batch_size
    N_train = len(dataset["train datapoints"])
    N_test = len(dataset["test datapoints"])
    loss_weights = loss_settings["weights"]

    optimizer = torch.optim.Adam(network.parameters(), lr=float(optimizer_settings["learning rate"]))

    train_losses = {
        "relation loss": [],
    }
    test_losses = []
    best_dice = 0
    relnet = network
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

            optimizer.step()

        train_loss = {
            "relation loss": loss_sum/n_batches,
            "sparse reg": sparse_reg.item(),
            "smooth reg": smooth_reg.item(),
        }
        wandb.log(train_loss)

        if epoch % optimizer_settings["validation frequency"] == 0:
            network.eval()
            wab.log_relnet(network, dataset)
            network.train()


def train_unet(paths, loss_settings, optimizer_settings, network, dataloaders, dataset):
    batch_size = dataloaders["train"].batch_size
    N_train = len(dataset["train datapoints"])
    N_test = len(dataset["test datapoints"])
    bce_loss = losses.get_bce_loss(dataset)

    optimizer = torch.optim.Adam(network.parameters(), lr=float(optimizer_settings["learning rate"]))

    train_losses = {
        "bce": [],
    }
    test_losses = []
    best_dice = 0
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

        train_loss = {
            "bce": loss_sum/n_batches,
        }
        wandb.log(train_loss)
        for k,v in train_loss.items():
            train_losses[k].append(v)

        if epoch % optimizer_settings["validation frequency"] == 0:
            network.eval()
            Y_hat = torch.sigmoid(Y_logits)
            wab.log_sample_outputs(X[0], Y_gt[0], Y_hat[0], dataset=dataset, name="training 1")
            wab.log_sample_outputs(X[1], Y_gt[1], Y_hat[1], dataset=dataset, name="training 2")

            loss_sum = 0
            n_batches = math.ceil(N_test/batch_size)

            # columns = ["image"]
            # if dataset["name"] != "MNIST grid":
            #     raise NotImplementedError("labels_to_record for non MNIST")
            # for label in mnist.labels_to_record:
            #     columns += [f"GT seg ({label})", f"predicted seg ({label})", f"IOU ({label})"]
            # table = wandb.Table(columns=columns)

            dices = []
            for batch in dataloaders["test"]:
                X,Y_gt = batch
                Y_logits = network(X)
                loss = bce_loss(Y_logits, Y_gt)
                loss_sum += loss.item()
                Y_hat = torch.sigmoid(Y_logits)

                dices.append(util.dice(Y_hat > .5, Y_gt > .5))

                # img = wab.to_wandb_img(X[0])
                # row = [img]
                # if dataset["name"] != "MNIST grid":
                #     raise NotImplementedError("labels_to_record for non MNIST")
                # for label in mnist.labels_to_record:
                #     legend = mnist.get_multi_mnist_legend()
                #     label_ix = legend[label]
                #     gt_seg = wab.to_wandb_img(Y_gt[0,label_ix])
                #     pred_seg = wab.to_wandb_img(Y_hat[0,label_ix])
                #     iou = util.iou(pred_seg > .5, gt_seg > .5)
                #     row += [gt_seg, pred_seg, iou]
                # table.add_data(*row)

            mean_dice = torch.cat(dices, dim=0).mean()
            if mean_dice > best_dice:
                run.summary["best_dice"] = best_dice = mean_dice

            test_loss = loss_sum/n_batches
            test_losses.append(test_loss)
            wandb.log({"test loss": test_loss})
            wab.log_sample_outputs(X[0], Y_gt[0], Y_hat[0], dataset=dataset, name="validation 1")
            wab.log_sample_outputs(X[1], Y_gt[1], Y_hat[1], dataset=dataset, name="validation 2")
            network.train()

        # if epoch % optimizer_settings["checkpoint frequency"] == 0:
        #     torch.save(network.state_dict(),
        #         os.path.join(paths["model weights directory"], f"{epoch}.state"))
        #     util.save_binary_file(data=network,
        #         path=os.path.join(paths["model weights directory"], f"{epoch}.inst"))

    # losses = {"train losses":train_losses, "test losses":test_losses}
    # torch.save(network.state_dict(), os.path.join(paths["model weights directory"], f"{epoch}.state"))
    # util.save_binary_file(data=network, path=os.path.join(paths["model weights directory"], f"{epoch}.inst"))
    # results = {"dataset": dataset, "model":network.eval(), "losses": losses}
    # util.save_binary_file(data=results, path=os.path.join("network histories folder", f"{job_name}.bin"))
    # return results


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
    # paths["model weights directory"] = os.path.join(paths["job output directory"], "models")
    # if not os.path.exists(paths["model weights directory"]):
    #     os.makedirs(paths["model weights directory"])

    for k in ("learning rate", "D learning rate", "G learning rate"):
        if k in args["optimizer"]:
            args["optimizer"][k] = float(args["optimizer"][k])

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
