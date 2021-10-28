import os, wandb
import numpy as np
import torch
nn = torch.nn
F = nn.functional

from sprelnet import util

def draw_images_with_colorbar():
    return

def lineplot():
    return

def plot_losses_for_spatial_relation_net_job(job):
    losses = util.get_losses_for_spatial_relation_net_job(job)
    labels = list(losses["train losses"].keys())
    lines = [losses["train losses"][k] for k in labels] + [losses["test losses"]]
    lineplot(lines, labels=labels+["test"])

def run_iterative_sprelnet_on_dp(seg, dp, n_iters=1000, dataset=None):
    if dataset is None:
        dataset = mnist.get_multi_mnist()
    X, Y_gt = dataset["datapoint loader"](dp)
    seg.eval()
    fixY = lambda y: torch.sigmoid(y.detach()).squeeze(0).cpu()
    X.unsqueeze_(0)
    Y_0 = seg.init_guess(X)
    y_0 = fixY(Y_0)
    Y = seg(X, Y_0=Y_0, n_iters=n_iters)

    return {"gt": (X.cpu(), Y_gt.cpu()),
        "outputs": (y_0, fixY(Y))}

def visualize_iterative_relations(seg, label1, label2, vmax=None, dataset=None):
    kernels = []
    for r_m in seg.rel_pyr.pyramid:
        W = r_m.weight.clone().cpu().detach()
        kernels.append(W[label1, label2])
    if vmax is None:
        draw_images_with_colorbar(kernels, padding=1)
    else:
        draw_images_with_colorbar(kernels, padding=1, vmax=vmax, vmin=-vmax)

    if dataset is not None:
        label_names = list(dataset["train label counts"].keys())
        A("set plot title")(f'"{label_names[label1]}" is more likely if "{label_names[label2]}"\nis observed in this relative position', case="lower")
    return kernels

def visualize_iterative_sprelnet_for_datapoint(seg, dp, iters=(100,500,1000),
        label_to_draw=1, dataset=None):
    X, Y_gt = dataset["datapoint loader"](dp)
    X.unsqueeze_(0)
    seg.eval()
    fixY = lambda y: torch.sigmoid(y.detach()).squeeze(0).cpu()
    Y_0 = seg.init_guess(X)
    Y = nn.Parameter(Y_0)
    optim = torch.optim.Adam([Y], lr=.05)
    best_Y = Y_0
    best_loss = np.inf
    Y_0 = fixY(torch.clone(Y_0))
    Ys = []
    losses = []
    for cur_iter in range(iters[-1]):
        loss = seg.get_test_loss(X, Y)
        if best_loss > loss.item():
            best_loss = loss.item()
            best_Y = Y.data
        if cur_iter+1 in iters:
            Ys.append(fixY(Y.data))
            losses.append(loss.item())
        loss.backward()
        optim.step()
    Ys = (Y_0, *Ys, fixY(best_Y))
    losses.append(best_loss)
    draw_images_with_colorbar([y[label_to_draw] for y in Ys])
    return {"gt": (X.cpu(), Y_gt.cpu()),
        "outputs": Ys,
        "losses": losses}
