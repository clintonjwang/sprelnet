import torch
nn = torch.nn
F = nn.functional

from sprelnet import util
from sprelnet.networks.relations import *
from sprelnet.networks.unet import *

# def randomly_scale_image():
# def randomly_rotate_image():
    
def get_contra_net(net_HPs, dataset):
    kwargs = {
        "image_size": dataset["image size"],
        "num_labels": util.get_num_labels(dataset),
        "kernel_size": (net_HPs["relation kernel size"], net_HPs["relation kernel size"]),
        "num_relations": net_HPs["number of relations"],
        "initseg_HPs": net_HPs["segmenter architecture"],
    }
    initseg_HPs = kwargs["initseg_HPs"]
    if isinstance(initseg_HPs["channels by depth"], str):
        initseg_HPs["channels by depth"] = initseg_HPs["channels by depth"].replace("N_L", str(kwargs["num_labels"]))
        for k,v in initseg_HPs.items():
            initseg_HPs[k] = util.parse_int_list(v)

    return ContrastiveSpRelNet(**kwargs).cuda()


# contrastive losses with test-time gradient ascent
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
                y = util.randomly_scale_image(y, weight)
            elif reg_type == "rotation robustness":
                if not hasattr(weight, "__iter__"):
                    weight = (-weight, weight)
                y = util.randomly_rotate_image(y, degree_range=weight)
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
