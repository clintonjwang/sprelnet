import torch
nn = torch.nn
F = torch.nn.functional

def determine_loss_weight(loss_name, epoch, loss_settings):
    base_weight = loss_settings["weights"][loss_name]
    if "ramp epochs" in loss_settings and loss_name in loss_settings["ramp epochs"]:
        return base_weight * min(epoch/loss_settings["ramp epochs"][loss_name], 1)
    else:
        return base_weight

def get_single_scale_kernel_regs(network, loss_weights):
    W = network.r_m.weight.clone()
    sparse_reg = (W.abs()+1e-5).sqrt().mean() * loss_weights["relation sparsity"]
    smooth_reg = ((W[...,1:,:] - W[...,:-1,:]).abs().mean() + (
        W[...,1:] - W[...,:-1]).abs().mean()) * loss_weights["relation smooth"]

    return sparse_reg, smooth_reg


def get_multiscale_kernel_regs(relnet, loss_weights):
    # regularize each kernel in the pyramid to be smooth and sparse
    sparse_reg = torch.zeros([]).cuda()
    smooth_reg = torch.zeros([]).cuda()
    for r_m in relnet.pyramid:
        W = r_m.weight.clone()
        sparse_reg += (W.abs()+1e-5).mean() * loss_weights["relation sparsity"]
        smooth_reg += ((W[...,1:,:] - W[...,:-1,:]).abs().mean() + (
            W[...,1:] - W[...,:-1]).abs().mean()) * loss_weights["relation smooth"]

    return sparse_reg, smooth_reg


def get_bce_loss(dataset):
    if dataset["name"] == "MNIST grid":
        N_train = len(dataset["train datapoints"])
        pos_weight = [N_train / cnt * dataset["digits per image"] for \
            label,cnt in dataset["train label counts"].items()]
    elif dataset["name"] == "pixels":
        pos_weight = [200.,300.,300.]
    else:
        raise NotImplementedError

    bce_fxn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).cuda())
    # transpose because pos_weight only works on the last dim
    return lambda y1,y2: bce_fxn(y1.transpose(1,-1), y2.transpose(1,-1))

def iou(pred_seg, gt_seg):
    with torch.no_grad():
        return ((pred_seg & gt_seg).sum(axis=(1,2)) / (pred_seg | gt_seg).sum(axis=(1,2)))

def dice(pred_seg, gt_seg):
    with torch.no_grad():
        return (2*(pred_seg & gt_seg).sum(axis=(1,2)) / (
            pred_seg.sum(axis=(1,2)) + gt_seg.sum(axis=(1,2))))
