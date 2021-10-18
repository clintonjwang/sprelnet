import math
import numpy as np
import torch
nn = torch.nn
F = nn.functional

from am.world.vision.affine import randomly_scale_image, randomly_rotate_image
from .unet import *



class match_XY(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.layers = nn.Sequential(DoubleConv(num_channels, 64), Down(64, 128),
            Down(128, 256), Down(256, 512), Down(512, 512),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512,1))

    def forward(self, x,y):
        return self.layers(torch.cat([x,y], dim=1)).squeeze(1)


class relation_pyramid(nn.Module): #multiscale
    def __init__(self, kernel_size, num_labels, N_levels):
        super().__init__()
        self.N_L = num_labels
        self.N_levels = N_levels #relations per label
        self.kernel_size = kernel_size

        self.pyramid = [nn.Conv2d(self.N_L, self.N_L, kernel_size=self.kernel_size,
                padding=[k//2 for k in self.kernel_size], bias=False).cuda() for _ in range(N_levels)]
        K = [k//2 for k in self.kernel_size]
        for r_m in self.pyramid:
            torch.nn.init.normal_(r_m.weight, std=1.) # aggressive initialization
            W = r_m.weight.data
            for i in range(self.N_L):
                W[i, i, K[0],K[1]].zero_() # "mask" the inpainted patch
            r_m.weight = nn.Parameter(W)

        self.bce = nn.BCEWithLogitsLoss()

    def score_Y(self, y):
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
            #score += -self.bce(logits.pow(3), y) #csprel5
            y = F.avg_pool2d(y,2)
        return score

    def forward(self, y):
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



class DirectSpRelNet(nn.Module):
    def __init__(self, image_size, num_labels, kernel_size=(9,9),
            num_relations=4, miniseg_HPs=None, multiscale=True):
        super().__init__()
        self.N_r = num_relations #relations per label
        self.N_L = num_labels
        self.kernel_size = kernel_size
        self.image_size = image_size
        self.D_XY = match_XY(num_labels + 1)

        self.annealing = annealing
        self.multiscale = multiscale

        if self.multiscale is True:
            self.relnet = relation_pyramid(kernel_size, num_labels, N_levels=num_relations)
        else:
            self.relnet = Id_Y(kernel_size, num_labels, num_relations)

        self.regularizers = {"total variation": 1.,
            "scaling robustness": 1.05,
            "rotation robustness": 5}

        if miniseg_HPs is None:
            self.miniseg = MiniSeg()
        else:
            self.miniseg = MiniSeg(miniseg_HPs["channels by depth"],
                    miniseg_HPs["kernels by depth"], miniseg_HPs["pool depths"])

    def score_Y(self, y):
        if self.multiscale is True:
            return self.relnet.score_Y(y)
        else:
            y_down = F.avg_pool2d(y,2)
            return -((torch.sigmoid(self.relnet(y_down)) - y_down).abs().sum([1,2,3]))

    def forward(self, X):
        return self.miniseg(X)


class ContrastiveSpRelNet(nn.Module):
    def __init__(self, image_size, num_labels, kernel_size=(9,9),
            num_relations=4, miniseg_HPs=None, multiscale=True, annealing=True):
        super().__init__()
        self.N_r = num_relations #relations per label
        self.N_L = num_labels
        self.kernel_size = kernel_size
        self.image_size = image_size
        self.match_XY = match_XY(num_labels + 1)

        self.annealing = annealing
        self.multiscale = multiscale

        if self.multiscale is True:
            self.rel_pyr = relation_pyramid(kernel_size, num_labels, N_levels=num_relations)
        else:
            self.Id_Y = Id_Y(kernel_size, num_labels, num_relations)

        if self.annealing is True:
            self.temperature = 100.
        else:
            self.temperature = 1.

        self.regularizers = {"total variation": 1.,
            "scaling robustness": 1.05,
            "rotation robustness": 5}

        if miniseg_HPs is None:
            self.init_guess = MiniSeg()
        else:
            self.init_guess = MiniSeg(miniseg_HPs["channels by depth"],
                    miniseg_HPs["kernels by depth"], miniseg_HPs["pool depths"])

    def score_Y(self, y):
        if self.multiscale is True:
            return self.rel_pyr.score_Y(y)
        else:
            y_down = F.avg_pool2d(y,2)
            return -((torch.sigmoid(self.Id_Y(y_down)) - y_down).abs().sum([1,2,3]))


    def get_test_loss(self, x, y):
        reg = torch.zeros([]).to(x.device)
        for reg_type, weight in self.regularizers.items():
            if reg_type == "total variation":
                loss = (y[...,1:,:] - y[...,:-1,:]).abs().mean() + (
                    y[...,1:] - y[...,:-1]).abs().mean()
                reg += loss * weight
            elif reg_type == "scaling robustness":
                if isinstance(weight, float):
                    weight = (1/weight, weight)
                y = randomly_scale_image(y, weight)
            elif reg_type == "rotation robustness":
                if not hasattr(weight, "__iter__"):
                    weight = (-weight, weight)
                y = randomly_rotate_image(y, degree_range=weight)
            else:
                raise NotImplementedError(f"bad regularizer type {reg_type}")

        pX_Y_loss = -self.match_XY(y, x).mean()
        pY_loss = -self.score_Y(y).mean()
        return pY_loss + pX_Y_loss + reg



    def forward(self, X, Y_0=None, n_iters=100):
        if Y_0 is None:
            Y_0 = self.init_guess(X)
        Y = nn.Parameter(Y_0)
        optim = torch.optim.Adam([Y], lr=.05)
        best_Y = Y_0
        best_loss = np.inf
        for _ in range(n_iters):
            loss = self.get_test_loss(X, Y)
            if best_loss > loss.item():
                best_loss = loss.item()
                best_Y = Y.data
            loss.backward()
            optim.step()
        return best_Y
