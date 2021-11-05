import os, wandb
import numpy as np
import torch
nn = torch.nn
F = nn.functional

from sprelnet import util

# def draw_image_with_colorbar(image, return_array=False, padding=None, pad_value=0, **kwargs):
#     fig,ax = plt.subplots()
#     plt.imshow(image, **kwargs)
#     plt.colorbar()
#     ax.set_axis_off()
#     if return_array is True:
#         array = get_figure_as_array(fig, height=pixels.height)
#         fig.set_visible(False)
#         return array
#     return ax

def get_multiscale_kernel_composite(relnet, label1, label2):
    kernels = []
    for lvl, r_m in enumerate(relnet.pyramid):
        W = r_m.weight.clone().cpu().detach()
        K = F.interpolate(W[label1:label1+1, label2:label2+1],
            scale_factor=(2.**lvl, 2.**lvl), mode="nearest").squeeze()
        kernels.append(K)
    w,h = max_size = K.shape
    composite = K

    for lvl in range(1, len(kernels)):
        d = 2**(lvl+1)
        di = 2**lvl - 1
        lw, lh = (di*w)//d, (di*h)//d
        dw, dh = kernels[-lvl-1].shape
        composite[lw:lw+dw, lh:lh+dh] += kernels[-lvl-1]

    return composite


def get_multiscale_likelihood_kernel_composite(relnet, label1, label2):
    # flatten/stack the kernel pyramid at the appropriate scale
    mean_kernels, stdev_kernels = [], []
    for lvl, r_m in enumerate(relnet.pyramid):
        W = r_m.weight.clone().cpu().detach()
        if len(W.shape) != 4:
            raise NotImplementedError(f"bad W shape {W.shape}")
        N_L = W.size(1)
        if label1 < 0:
            label1 += N_L
        if label2 < 0:
            label2 += N_L
        mK = F.interpolate(W[label1:label1+1, label2:label2+1],
            scale_factor=(2.**lvl, 2.**lvl), mode="nearest").squeeze()
        sK = F.interpolate(W[label1+N_L:label1+N_L+1, label2:label2+1],
            scale_factor=(2.**lvl, 2.**lvl), mode="nearest").squeeze()
        mean_kernels.append(mK)
        stdev_kernels.append(sK)

    try:
        w,h = max_size = mK.shape
    except ValueError:
        raise NotImplementedError(f"bad mK shape {mK.shape} (W shape {W.shape})")

    mean_composite = mK
    stdev_composite = sK
    for lvl in range(1, len(mean_kernels)):
        d = 2**(lvl+1)
        di = 2**lvl - 1
        lw, lh = (di*w)//d, (di*h)//d
        dw, dh = mean_kernels[-lvl-1].shape
        mean_composite[lw:lw+dw, lh:lh+dh] += mean_kernels[-lvl-1]
        stdev_composite[lw:lw+dw, lh:lh+dh] += stdev_kernels[-lvl-1]

    return mean_composite, stdev_composite



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

# def visualize_relations(seg, label1, label2, vmax=None, dataset=None):
#     kernels = []
#     for r_m in seg.rel_pyr.pyramid:
#         W = r_m.weight.clone().cpu().detach()
#         kernels.append(W[label1, label2])
#     if vmax is None:
#         draw_images_with_colorbar(kernels, padding=1)
#     else:
#         draw_images_with_colorbar(kernels, padding=1, vmax=vmax, vmin=-vmax)

#     if dataset is not None:
#         if dataset["name"] == "MNIST grid":
#             label_names = list(dataset["train label counts"].keys())
#             A("set plot title")(f'"{label_names[label1]}" is more likely if "{label_names[label2]}"\nis observed in this relative position', case="lower")
#         else:
#             print("TBD")
#     return kernels


# def visualize_iterative_sprelnet_for_datapoint(seg, dp, iters=(100,500,1000),
#         label_to_draw=1, dataset=None):
#     X, Y_gt = dataset["datapoint loader"](dp)
#     X.unsqueeze_(0)
#     seg.eval()
#     fixY = lambda y: torch.sigmoid(y.detach()).squeeze(0).cpu()
#     Y_0 = seg.init_guess(X)
#     Y = nn.Parameter(Y_0)
#     optim = torch.optim.Adam([Y], lr=.05)
#     best_Y = Y_0
#     best_loss = np.inf
#     Y_0 = fixY(torch.clone(Y_0))
#     Ys = []
#     losses = []
#     for cur_iter in range(iters[-1]):
#         loss = seg.get_test_loss(X, Y)
#         if best_loss > loss.item():
#             best_loss = loss.item()
#             best_Y = Y.data
#         if cur_iter+1 in iters:
#             Ys.append(fixY(Y.data))
#             losses.append(loss.item())
#         loss.backward()
#         optim.step()
#     Ys = (Y_0, *Ys, fixY(best_Y))
#     losses.append(best_loss)
#     draw_images_with_colorbar([y[label_to_draw] for y in Ys])
#     return {"gt": (X.cpu(), Y_gt.cpu()),
#         "outputs": Ys,
#         "losses": losses}
