import argparse, itertools, os, shutil, yaml, math
import numpy as np
import torch
nn = torch.nn
F = nn.functional

torch.backends.cudnn.deterministic = True

from sprelnet import util, mnist, wab
from sprelnet.networks.sprelnet import get_adv_sprelnet, get_relnet
from sprelnet.networks.unet import get_unet

import wandb


def train_sprelnet(paths, loss_settings, optimizer_settings, network, dataloaders, dataset):
    G = network.G
    relnet = network.relnet
    if network.type == "adversarial":
        D = network.D_XY
    loss_weights = loss_settings["weights"]

    # setup dataloaders
    batch_size = dataloaders["train"].batch_size
    N_train = len(dataset["train datapoints"])
    N_test = len(dataset["test datapoints"])
    pos_weight = [N_train / cnt * dataset["digits per image"] for \
        label,cnt in dataset["train label counts"].items()]
    bce_fxn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).cuda())
    # transpose because pos_weight only works on the last dim
    bce_loss = lambda y1,y2: bce_fxn(y1.transpose(1,-1), y2.transpose(1,-1))

    G_optim = torch.optim.Adam(G.parameters(), lr=float(optimizer_settings["G learning rate"]))
    if network.type == "adversarial":
        DR_optim = torch.optim.Adam(itertools.chain(relnet.parameters(), D.parameters()),
            lr=float(optimizer_settings["D learning rate"]))
    else:
        DR_optim = torch.optim.Adam(relnet.parameters(), lr=float(optimizer_settings["D learning rate"]))

    loss_names = ["rel true", "rel fake", "seg loss",
        "sparse kernel regularization", "smooth kernel regularization"]
    if network.type == "adversarial":
        loss_names += ["-log pxy true", "-log pxy fake"]

    train_losses = {
        k:[] for k in loss_names
    }
    test_losses = []

    # static regularizers
    sparse_reg = torch.zeros([]).cuda()
    smooth_reg = torch.zeros([]).cuda()
    K = [k//2 for k in network.kernel_size]
    for r_m in relnet.pyramid:
        W = r_m.weight.clone()
        sparse_reg += (W.abs()+1e-5).mean() * loss_weights["relation sparsity"]
        smooth_reg += ((W[...,1:,:] - W[...,:-1,:]).abs().mean() + (
            W[...,1:] - W[...,:-1]).abs().mean()) * loss_weights["relation smooth"]

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
                G_loss = seg_loss - p_xy_fake + rel_fake
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
            sparse_reg.backward(retain_graph=True)
            smooth_reg.backward(retain_graph=True)
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
            network.eval()
            loss_sum = 0
            n_batches = math.ceil(N_test/batch_size)
            for batch in dataloaders["test"]:
                X,Y_gt = batch
                Y_hat = G(X)
                loss = bce_loss(Y_hat, Y_gt)
                loss_sum += loss.item()
            test_loss = loss_sum/n_batches

            wandb.log({"test loss": test_loss})

            network.train()

        if epoch % optimizer_settings["checkpoint frequency"] == 0:
            torch.save(network.state_dict(),
                os.path.join(paths["model weights directory"], f"{epoch}.state"))
            util.save_binary_file(data=network,
                path=os.path.join(paths["model weights directory"], f"{epoch}.inst"))

    #wandb.watch(network)

    losses = {"train losses":train_losses, "test losses":test_losses}
    torch.save(network.state_dict(), os.path.join(paths["model weights directory"], f"{epoch}.state"))
    util.save_binary_file(data={"train losses":train_losses, "test losses":test_losses},
        path=paths["loss history path"])
    util.save_binary_file(data=network, path=os.path.join(paths["model weights directory"], f"{epoch}.inst"))
    return results


def train_unet(paths, loss_settings, optimizer_settings, network, dataloaders, dataset):
    batch_size = dataloaders["train"].batch_size
    N_train = len(dataset["train datapoints"])
    N_test = len(dataset["test datapoints"])
    pos_weight = [N_train / cnt * dataset["digits per image"] for \
        label,cnt in dataset["train label counts"].items()]
    bce_fxn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).cuda())
    # transpose because pos_weight only works on the last dim
    bce_loss = lambda y1,y2: bce_fxn(y1.transpose(1,-1), y2.transpose(1,-1))
    labels_to_record = ["6 left of 7", "7 right of 7", "all 7s"]

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
            Y_hat = network(X)
            loss = bce_loss(Y_hat, Y_gt)
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
            loss_sum = 0
            n_batches = math.ceil(N_test/batch_size)

            columns = ["image"]
            for label in labels_to_record:
                columns += [f"GT seg ({label})", f"predicted seg ({label})", f"IOU ({label})"]
            table = wandb.Table(columns=columns)

            for batch in dataloaders["test"]:
                X,Y_gt = batch
                Y_logits = network(X)
                loss = bce_loss(Y_logits, Y_gt)
                loss_sum += loss.item()
                Y_hat = torch.sigmoid(Y_logits)

                dice = util.dice(Y_hat > .5, Y_gt > .5)

                img = wab.to_wandb_img(X[0])
                row = [img]
                for label in labels_to_record:
                    legend = mnist.get_multi_mnist_legend()
                    label_ix = legend[label]
                    gt_seg = wab.to_wandb_img(Y_gt[0,label_ix])
                    pred_seg = wab.to_wandb_img(Y_hat[0,label_ix])
                    iou = util.iou(pred_seg > .5, gt_seg > .5)
                    row += [gt_seg, pred_seg, iou]
                table.add_data(*row)

            if dice > best_dice:
                run.summary["best_dice"] = best_dice = dice

            test_loss = loss_sum/n_batches
            test_losses.append(test_loss)
            wandb.log({"gt_seg": gt_seg, "pred_seg": pred_seg, "test loss": test_loss})
            network.train()

        if epoch % optimizer_settings["checkpoint frequency"] == 0:
            torch.save(network.state_dict(),
                os.path.join(paths["model weights directory"], f"{epoch}.state"))
            util.save_binary_file(data=network,
                path=os.path.join(paths["model weights directory"], f"{epoch}.inst"))

    losses = {"train losses":train_losses, "test losses":test_losses}
    torch.save(network.state_dict(), os.path.join(paths["model weights directory"], f"{epoch}.state"))
    util.save_binary_file(data=network, path=os.path.join(paths["model weights directory"], f"{epoch}.inst"))
    # results = {"dataset": dataset, "model":network.eval(), "losses": losses}
    # util.save_binary_file(data=results, path=os.path.join("network histories folder", f"{job_name}.bin"))
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path')
    parser.add_argument('--job_id', default='test_00')
    cmd_args = parser.parse_args()
    # class cmd_args:
    #     job_id = "test_00"
    #     config_path = f"/data/vision/polina/users/clintonw/code/sprelnet/configs/{job_id}.yaml"
    with open(cmd_args.config_path, 'r') as stream:
        args = yaml.safe_load(stream)
        
    if "tags" not in args:
        args["tags"] = [args["network"]["type"]]

    config = {k:args[k] for k in ["network", "loss", "optimizer", "data loading"]}
    run = wandb.init(project="spatial_relations", config=config, tags=args["tags"],
            name=cmd_args.job_id) #entity="clintonjwang",
    run_id = os.path.basename(run.path)

    # fill in paths
    paths = args["paths"]
    paths["job output directory"] = util.get_job_output_folder(cmd_args.job_id)
    # paths["loss history path"] = os.path.join(paths["job output directory"], "losses.bin")
    paths["model weights directory"] = os.path.join(paths["job output directory"], "models")
    if not os.path.exists(paths["model weights directory"]):
        os.makedirs(paths["model weights directory"])

    for k in ("learning rate", "D learning rate", "G learning rate"):
        if k in args["optimizer"]:
            args["optimizer"][k] = float(args["optimizer"][k])

    # copy code base and config file to slurm output
    util.save_code_dir(util.get_code_directory(), paths["job output directory"])
    config_target_path = os.path.join(paths["job output directory"], "config.yaml")
    shutil.copy(cmd_args.config_path, config_target_path)
    with open(config_target_path, "a") as f:
        f.write(f"\nrun id: {run_id}")

    # set random seed
    np.random.seed(args["random seed"])
    torch.manual_seed(args["random seed"])
    if args["fixed input size"] is True:
        torch.backends.cudnn.benchmark = True

    # if dataset == "MNIST grid":
    #     dataset = mnist.get_multi_mnist()
    dataset = mnist.get_multi_mnist()
    dataloaders = util.get_dataloaders(dataset, batch_size=args["data loading"]["batch size"])

    if args["network"]["type"] in ["adversarial", "vanilla"]:
        network = get_adv_sprelnet(net_HPs=args["network"], dataset=dataset)

        train_sprelnet(paths=args["paths"],
            loss_settings=args["loss"], optimizer_settings=args["optimizer"],
            dataloaders=dataloaders, network=network, dataset=dataset)

    elif args["network"]["type"] == "unet":
        network = get_unet(net_HPs=args["network"], dataset=dataset)

        train_unet(paths=args["paths"],
            loss_settings=args["loss"], optimizer_settings=args["optimizer"],
            dataloaders=dataloaders, network=network, dataset=dataset)

    elif args["network"]["type"] == "relations":
        network = get_relnet(net_HPs=args["network"], dataset=dataset)

        train_unet(paths=args["paths"],
            loss_settings=args["loss"], optimizer_settings=args["optimizer"],
            dataloaders=dataloaders, network=network, dataset=dataset)

    else:
        raise NotImplementedError
