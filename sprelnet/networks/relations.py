import math
import numpy as np
import torch
nn = torch.nn
F = nn.functional

from sprelnet import util
from sprelnet.networks.unet import *

def get_relnet(net_HPs, dataset):
    kernel_size = (net_HPs["relation kernel size"], net_HPs["relation kernel size"])
    num_labels = util.get_num_labels(dataset)
    N_levels = net_HPs["number of relations"]

    return rel_pyramid_likelihood(kernel_size, num_labels, N_levels)


class match_XY(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.layers = nn.Sequential(DoubleConv(num_channels, 64), Down(64, 128),
            Down(128, 256), Down(256, 512), Down(512, 512),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512,1))

    def forward(self, x,y):
        return self.layers(torch.cat([x,y], dim=1)).squeeze(1)

class rel_pyramid_likelihood(nn.Module):
    def __init__(self, kernel_size, num_labels, N_levels):
        super().__init__()
        self.N_L = num_labels
        self.N_levels = N_levels #relations per label
        self.kernel_size = kernel_size

        K = [k//2 for k in self.kernel_size]
        self.pyramid = nn.Sequential(*[nn.Conv2d(self.N_L, 2*self.N_L, kernel_size=self.kernel_size,
                padding=K, bias=False).cuda() for _ in range(N_levels)])
        for r_m in self.pyramid:
            #torch.nn.init.normal_(r_m.weight, std=1.) # aggressive initialization
            W = r_m.weight.data
            for i in range(self.N_L):
                W[i:W.size(0):self.N_L, i, K[0],K[1]].zero_() # "mask" the inpainted patch
            r_m.weight = nn.Parameter(W)

    def forward(self, y, eps=1e-6):
        # should return a loss (to be minimized individually and for the generator)
        nll = torch.zeros(y.size(0)).to(y.device)
        for level in range(self.N_levels):
            r_m = self.pyramid[level](y).view(-1, 2, self.N_L, y.size(-2), y.size(-1))
            mu,var = r_m[:,0], torch.clamp(torch.exp(r_m[:,1]), min=eps)
            nll += ((y - mu).pow(2) / var + torch.log(var)).sum([1,2,3]).mean(0)
            y = F.avg_pool2d(y,2)
        return nll

    def get_pyramid(self, y):
        outputs = []
        for level in range(self.N_levels):
            outputs.append(self.pyramid[level](y))
            y = F.avg_pool2d(y,2)
        return outputs


class rel_pyramid_estimator(nn.Module):
    def __init__(self, kernel_size, num_labels, N_levels):
        super().__init__()
        self.N_L = num_labels
        self.N_levels = N_levels #relations per label
        self.kernel_size = kernel_size

        K = [k//2 for k in self.kernel_size]
        self.pyramid = nn.Sequential(*[nn.Conv2d(self.N_L, self.N_L, kernel_size=self.kernel_size,
                padding=K, bias=False).cuda() for _ in range(N_levels)])
        for r_m in self.pyramid:
            #torch.nn.init.normal_(r_m.weight, std=1.) # aggressive initialization
            W = r_m.weight.data
            for i in range(self.N_L):
                W[i, i, K[0],K[1]].zero_() # "mask" the inpainted patch
            r_m.weight = nn.Parameter(W)

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y):
        # should return a loss (to be minimized individually and for the generator)
        # we do not require every relation to match in order to have a high score,
        # matching a few relations well should give a better score...
        # weight different levels differently?
        # laplacian pyramid instead? (use the residues in each level)
        # add confidence as auxiliary output to the kernel?
        score = torch.zeros(y.size(0)).to(y.device)
        for level in range(self.N_levels):
            logits = self.pyramid[level](y)
            y_hat = torch.sigmoid(logits)
            #score += 1 / (logits*(y_hat-y)).abs().mean([1,2,3]) #csprel2
            #score += 1 / (y_hat-y).abs().mean([1,2,3]) #csprel3
            score += (1 / (y_hat-y).abs().mean([2,3])).mean(1) #csprel1, csprel4
            #score += (1 / (logits*(y_hat-y)).abs().mean([2,3])).mean(1)
            #score += -self.bce(logits.pow(3), y) #csprel5
            y = F.avg_pool2d(y,2)
        return -score #torch.log

    def get_pyramid(self, y):
        outputs = []
        for level in range(self.N_levels):
            outputs.append(self.pyramid[level](y))
            y = F.avg_pool2d(y,2)
        return outputs


class Id_Y(nn.Module):
    def __init__(self, kernel_size, num_labels, num_relations):
        super().__init__()
        self.N_L = num_labels
        self.N_r = num_relations #relations per label
        self.kernel_size = kernel_size

        self.r_m = nn.Conv2d(self.N_L, self.N_r*self.N_L, kernel_size=self.kernel_size,
                padding=[k//2 for k in self.kernel_size], bias=False)
        K = [k//2 for k in self.kernel_size]
        torch.nn.init.normal_(self.r_m.weight, std=1.) # aggressive initialization
        W = self.r_m.weight.data
        for i in range(self.N_L):
            W[i:W.size(0):self.N_L, i, K[0],K[1]].zero_() # "mask" the inpainted patch
        self.r_m.weight = nn.Parameter(W)

    def forward(self, y):
        r_m = self.r_m(y).view(-1, self.N_r, self.N_L, y.size(-2), y.size(-1))
        return r_m.sum(1)

