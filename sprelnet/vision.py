import torch
nn = torch.nn
F = nn.functional

def get_height(image):
    if image.shape[0] > 3:
        return image.shape[0]
    else:
        return image.shape[1]


def apply_padding(tensor, pad, value=0):
    if len(tensor.shape) < 3:
        tensor.unsqueeze_(0)
    if len(tensor.shape) < 4:
        tensor.unsqueeze_(0)
    ddims = len(tensor.shape) - len(pad)//2
    for _ in range(ddims,2):
        tensor = tensor.unsqueeze(0)
    tensor = F.pad(tensor, pad, value=value)
    for _ in range(ddims,2):
        tensor = tensor.squeeze(0)
    return tensor

def pad_evenly_by_amount(tensor, padding, value=0):
    if len(tensor.shape) < 3:
        tensor.unsqueeze_(0)
    if len(tensor.shape) < 4:
        tensor.unsqueeze_(0)
    pad = []
    if not hasattr(padding, "__len__"):
        padding = [padding] * (len(tensor.shape)-2)
    for p in padding:
        pad = [p//2,(p+1)//2] + pad
    return apply_padding(tensor, pad, value=value)

def pad_evenly_to_shape(tensor, shape):
    ddims = len(tensor.shape)-len(shape)
    if ddims == 0:
        tensor.unsqueeze_(0)
        ddims = 1
    padding = [max(shape[i] - tensor.size(i+ddims), 0) for i in range(len(shape))]
    return pad_evenly_by_amount(tensor, padding)

def center_crop_to_shape(t, shape):
    ddims = len(t.shape)-len(shape)
    crops = [max(t.size(i+ddims) - shape[i], 0) for i in range(len(shape))]
    for ix,crop in enumerate(crops):
        t = torch.narrow(t, ddims+ix, crop//2, shape[ix])
    return t

def pad_or_crop_to_shape(tensor, shape):
    tensor = pad_evenly_to_shape(tensor, shape)
    return center_crop_to_shape(tensor, shape)
# def pad_tensors_to_match(t1, t2, n_dims=3):
#     ddims = len(t1.shape) - n_dims
#     if ddims == 0: raise ValueError("tensors need batch dim")
#     pad1 = [max(t2.size(i+ddims) - t1.size(i+ddims),0) for i in range(n_dims)]
#     pad2 = [max(t1.size(i+ddims) - t2.size(i+ddims),0) for i in range(n_dims)]
#     t1 = pad_evenly_by_amount(t1, pad1)
#     t2 = pad_evenly_by_amount(t2, pad2)
#     return t1, t2

def pad_to_multiple(x, patch_size):
    if len(patch_size) == 3:
        kc, kh, kw = patch_size
        pc, ph, pw = (-x.size(-3)%kc, -x.size(-2)%kh, -x.size(-1)%kw)
        padding = (pw//2, (pw+1)//2, ph//2, (ph+1)//2, pc//2, (pc+1)//2)

    elif len(patch_size) == 2:
        kh, kw = patch_size
        ph, pw = (-x.size(-2)%kh, -x.size(-1)%kw)
        padding = (pw//2, (pw+1)//2, ph//2, (ph+1)//2)

    else:
        raise NotImplementedError

    return F.pad(x, padding)

def pad_tensors_to_match(*tensors, n_dims=3):
    if len(tensors) == 1: tensors = tensors[0]
    ddims = len(tensors[0].shape) - n_dims
    spatial_dims = [max([img.size(i+ddims) for img in tensors]) for i in range(n_dims)]
    return [pad_evenly_to_shape(img, spatial_dims) for img in tensors]
