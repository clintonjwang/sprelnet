import torch
nn = torch.nn
F = torch.nn.functional

def get_multiscale_kernel_regs(relnet, loss_weights):
    sparse_reg = torch.zeros([]).cuda()
    smooth_reg = torch.zeros([]).cuda()
    K = [k//2 for k in relnet.kernel_size]
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
