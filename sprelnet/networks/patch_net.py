import math, einops
import numpy as np
import torch
nn = torch.nn
F = nn.functional

from sprelnet import util, vision
from sprelnet.networks.unet import MiniSeg

def get_patch_net(net_HPs, dataset):
    kwargs = {
        "image_size": dataset["image size"],
        "patch_size": util.parse_int_list(net_HPs["patch size"]),
        "num_labels": util.get_num_labels(dataset),
        "kernel_size": (net_HPs["relation kernel size"], net_HPs["relation kernel size"]),
        "num_heads": net_HPs["number of heads"],
        "num_relations": net_HPs["number of relations"],
        "initseg_HPs": net_HPs["segmenter architecture"],
        "template_HPs": net_HPs["PatchTemplate HPs"],
    }
    initseg_HPs = kwargs["initseg_HPs"]
    if isinstance(initseg_HPs["channels by depth"], str):
        initseg_HPs["channels by depth"] = initseg_HPs["channels by depth"].replace("N_L", str(kwargs["num_labels"]))
        for k,v in initseg_HPs.items():
            initseg_HPs[k] = util.parse_int_list(v)

    template_HPs = kwargs["template_HPs"]
    if isinstance(template_HPs["channels by depth"], str):
        template_HPs["channels by depth"] = template_HPs["channels by depth"].replace("N_L*N_V", str(kwargs["num_labels"]*kwargs["num_heads"]))
        for k,v in template_HPs.items():
            template_HPs[k] = util.parse_int_list(v)

    return IterPatchSpRelNet(**kwargs).cuda()



class PatchTemplate(nn.Module):
    def __init__(self, channels_by_depth, kernels_by_depth):
        super().__init__()
        n_layers = len(kernels_by_depth)
        assert n_layers == len(channels_by_depth) - 1, "#channels and #kernels misaligned"
        pad_by_depth = [(0, k//2, k//2) for k in kernels_by_depth]
        kernels_by_depth = [(1,k,k) for k in kernels_by_depth]

        self.layers = []
        for d in range(n_layers-1):
            self.layers += [nn.Conv3d(channels_by_depth[d], channels_by_depth[d+1],
                    kernel_size=kernels_by_depth[d], padding=pad_by_depth[d]),
                    nn.BatchNorm3d(channels_by_depth[d+1]), nn.ReLU()]

        self.layers.append(nn.Conv3d(channels_by_depth[-2], channels_by_depth[-1],
                    kernel_size=kernels_by_depth[-1], padding=pad_by_depth[-1]))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, X):
        return self.layers(X)


def torch_img_to_patch_2d(image, patch_size, flattened=False): # (batch, channel, x,y)
    x = vision.pad_to_multiple(image, patch_size)
    x = x.unfold(3, patch_size[1], patch_size[1]) # (batch, channel, x, y_token, y_pixel)
    x = x.unfold(2, patch_size[0], patch_size[0]) # (batch, channel, x_token, y_token, y_pixel, x_pixel)
    if flattened:
        patches = einops.rearrange(x, 'b c xt yt yp xp -> b c (xt yt) (xp yp)') # (batch, channel, token, pixel)
    else:
        patches = einops.rearrange(x, 'b c xt yt yp xp -> b c (xt yt) xp yp')
    # kh, kw = patch_size
    # x = x.unfold(-2, kh, kh).unfold(-1, kw, kw)
    # patches = x.contiguous().view(*image.shape[:-2], kh*kw)
    return patches

def torch_patch_to_img_2d(patches, image_size): # (batch, channel, token, patch_x, patch_y)
    patch_size = patches.shape[-2:]
    token_lens = [math.ceil(image_size[dim]/patch_size[dim]) for dim in range(2)]
    x = patches.view(*patches.shape[:-3], *token_lens, *patch_size)
    pad_image_size = [token_lens[dim]*patch_size[dim] for dim in range(2)]
    image = x.permute(0,1,2,4,3,5).reshape(*patches.shape[:-3], *pad_image_size)
    if image.shape[-2:] != image_size:
        image = vision.center_crop_to_shape(image, image_size)
    return image # (batch, channel, x,y)

def torch_flat_patch_to_img_2d(patches, image_size, patch_size=None): # (batch, channel, token, pixel)
    if patch_size is None:
        patch_size = [round(patches.size(-1)**.5)] * 2
    patches = patches.view(*patches.shape[:-1], *patch_size)
    return torch_patch_to_img_2d(patches, image_size)


# iterative, attention over "patch proposals"
class IterPatchSpRelNet(nn.Module):
    def __init__(self, image_size, num_labels, kernel_size=(9,9), num_heads=8,
            num_relations=12, patch_size=(8,8), initseg_HPs=None, template_HPs=None):
        super().__init__()
        if len(image_size) != 2:
            raise NotImplementedError
        d = [math.ceil(image_size[dim]/patch_size[dim]) for dim in range(2)]
        self.N_T = num_tokens = int(np.prod(d))
        self.N_V = num_heads # heads per label
        self.N_r = num_relations #relations per label
        self.patch_size = patch_size
        self.N_P = np.prod(patch_size)
        self.N_L = num_labels

        self.rel_kernel_size = kernel_size
        self.image_size = image_size

        if initseg_HPs is None:
            self.init_guess = MiniSeg().train().cuda()
        else:
            self.init_guess = MiniSeg(initseg_HPs["channels by depth"],
                    initseg_HPs["kernels by depth"], initseg_HPs["pool depths"]).train().cuda()

        if template_HPs is None:
            self.W_V = PatchTemplate().train().cuda()
        else:
            self.W_V = PatchTemplate(template_HPs["channels by depth"], template_HPs["kernels by depth"]).train().cuda()


        self.pool = nn.AvgPool2d(patch_size, ceil_mode=True) # this doesn't need to be so coarse?
        #self.r_m = SparseKernel()
        self.r_m = nn.Conv2d(self.N_L, self.N_r*self.N_L, kernel_size=kernel_size,
                padding=[k//2 for k in kernel_size], bias=False)
        K = [k//2 for k in self.rel_kernel_size]
        W = self.r_m.weight.data
        for i in range(self.N_L):
            W[i:W.size(0):self.N_L, i, K[0],K[1]].zero_() # "mask" the inpainted patch
        self.r_m.weight = nn.Parameter(W)

        self.tokenize = torch_img_to_patch_2d
        self.untokenize = torch_patch_to_img_2d


    def V(self, x): # (batch, 1, token, pixel)
        return self.W_V(x).reshape(-1, self.N_L, self.N_V, self.N_T, *self.patch_size).transpose(2,3)

    def r(self, Y_0):
        Y_down = self.pool(Y_0) # (batch, label, token_x, token_y)
        r_m = self.r_m(Y_down).view(-1, self.N_r, self.N_L,
                Y_down.size(-1)*Y_down.size(-2)) # (batch, relation, label, token)
        return r_m.sum(1) # (batch, label, token)

    def forward(self, X, Y_0=None): # (batch, 1, x,y)
        if Y_0 is None:
            Y_0 = self.init_guess(X) # (batch, label, x,y)
        r = self.r(Y_0) # (batch, label, token)
        x = self.tokenize(X, self.patch_size) # (batch, 1, token, tx,ty)
        Vx = self.V(x) # (batch, label, token, head, tx,ty)
        c = Vx.sum([-1,-2]) # (batch, label, token, head)
        Vx = torch.cat([Vx, torch.full_like(Vx[...,:1,:,:], -10.)], dim=-3) # add head with blank patch template
        a = torch.einsum('blth,blt->bth', c, r) # (batch, token, head)
        a = torch.cat([a, torch.zeros_like(a[...,:1])], dim=-1) # add 0 logit for the blank patch
        a = F.softmax(a, dim=-1)
        y = torch.einsum('bth,blthxy->bltxy', a, Vx) # (batch, label, token, tx,ty)
        Y = self.untokenize(y, image_size=self.image_size) # (batch, label, x,y)
        return Y

