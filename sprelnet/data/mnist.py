import os, wandb, shutil, cv2, json
import numpy as np
import torch
import torchvision.datasets
import skimage.io as io
from pycocotools.coco import COCO
import pycocotools.mask as cocomask
import matplotlib.pyplot as plt
from sprelnet import util
osp = os.path
F = torch.nn.functional

ds_folder = '/data/vision/polina/scratch/clintonw/datasets'
mnist_grid_dir = "/data/vision/polina/scratch/clintonw/datasets/mnist_grid"
mnist_ds_path = osp.join(ds_folder, "mnist.bin")
multi_mnist_ds_path = osp.join(ds_folder, 'mnist_grid/ds_instance.bin')

labels_to_record = ["7 left of 6", "7 right of 6", "all 6s"]
# label_pairs_of_interest = [(0,7),(0,0),(-1,-5)]
# pos_label_pairs=[(0,7),]
# id_label_pairs=[(0,0),]
# neg_label_pairs=[(-1,-5),]
# rand_label_pairs=[(0,1),(0,2)]

def get_mnist():
    dataset = torchvision.datasets.MNIST(ds_folder, train=True, download=True)
    train_images, train_labels = dataset.data, dataset.targets

    dataset = torchvision.datasets.MNIST(ds_folder, train=False, download=True)
    test_images, test_labels = dataset.data, dataset.targets

    return (train_images, train_labels), (test_images, test_labels)

def get_multi_mnist(run=None, **kwargs):
    if osp.exists(multi_mnist_ds_path):
        if run is not None:
            artifact = wandb.Artifact('training_data', type='dataset')
            artifact.add_file(multi_mnist_ds_path)
            run.log_artifact(artifact)
        return util.load_binary_file(multi_mnist_ds_path)
    else:
        return create_multi_MNIST_dataset(**kwargs)

def get_multi_mnist_legend(key="name", stage=1):
    relations = ["above", "left of", "right of", "below",
        #"imm. above", "imm. left of", "imm. right of", "imm. below",
        #"adjacent to", "horiz. next to", "vert. next to",
        #"between"
    ]
    if stage == 1:
        label_names = [f"7 {r} 6" for r in relations] + ["other 7s", "all 6s"]
    elif stage == 2:
        label_names = [f"7 {r} 6" for r in relations] + [f"7 {r} 7" for r in relations] + ["all 6s"]
        #label_names = ["6|", "6-", "6/", "6\\", "all 6s"]
    elif stage == 3:
        label_names = ["6 horizontal neighbors", "any col with a 6",
            "any / with a 6", "any \\ with a 6", "all 6s"]

    #"9 immediately above 7", "9 immediately left of 7",
    #"7 adjacent to 6", "3 adjacent to 4",
    if key == "name":
        return {name:ix for ix,name in enumerate(label_names)}
    elif key == "index" or key is None:
        return label_names
    elif key == "class labels":
        return {ix+1:name for ix,name in enumerate(label_names)}
    else:
        raise NotImplementedError

def get_annotation_json(phase="val"):
    return osp.join(mnist_grid_dir, "val/annotation_coco.json")

def mask2poly(mask):
    contours, _ = cv2.findContours(mask.numpy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for object in contours:
        coords = []
        for point in object:
            coords.append(float(point[0][0]))
            coords.append(float(point[0][1]))
        polygons.append(coords)
    return polygons

def create_mnist_grid_coco_dataset(N_train=4000, N_val=1000, digits_per_dim=(5,5), scale_factor=2):
    if digits_per_dim != (5,5):
        raise NotImplementedError
    if osp.exists(mnist_grid_dir):
        shutil.rmtree(mnist_grid_dir)
    os.makedirs(osp.join(mnist_grid_dir, "train"))
    os.makedirs(osp.join(mnist_grid_dir, "val"))

    train_mnist, test_mnist = get_mnist()
    base_dataset = {
        "images": torch.cat([train_mnist[0], test_mnist[0]], 0),
        "labels": torch.cat([train_mnist[1], test_mnist[1]], 0),
    }
    base_dataset["images by label"] = [base_dataset["images"][base_dataset["labels"] == ix] for ix in range(10)]

    category_names = list(get_multi_mnist_legend().keys())
    n_digits = np.prod(digits_per_dim)
    coco_categories = []
    for ix, label_name in enumerate(category_names):
        if label_name.startswith("all") or label_name.startswith("other"):
            coco_categories.append({
                "id": ix, "name": label_name, "supercategory": label_name[-2],
            })
        else:
            coco_categories.append({
                "id": ix, "name": label_name, "supercategory": label_name[0],
            })

    imgs = []
    annotations = []
    for dp_id in range(N_train):
        img_info, anns = create_mnist_grid_coco_datapoint(base_dataset, n_digits,
            dp_id, scale_factor=scale_factor, ann_id=len(annotations), img_dir=osp.join(mnist_grid_dir, "train"),
            category_names=category_names)
        imgs.append(img_info)
        annotations += anns
        if dp_id % 100 == 0: print(dp_id)
    train_ds = {
        "images": imgs,
        "annotations": annotations,
        "categories": coco_categories,
    }
    with open(osp.join(mnist_grid_dir, "train/annotation_coco.json"), 'w') as f:
        json.dump(train_ds, f)

    imgs = []
    annotations = []
    for dp_id in range(N_train, N_val+N_train):
        img_info, anns = create_mnist_grid_coco_datapoint(base_dataset, n_digits,
            dp_id, scale_factor=scale_factor, ann_id=len(annotations), img_dir=osp.join(mnist_grid_dir, "val"),
            category_names=category_names)
        imgs.append(img_info)
        annotations += anns
        if dp_id % 100 == 0: print(dp_id)
    val_ds = {
        "images": imgs,
        "annotations": annotations,
        "categories": coco_categories,
    }
    with open(osp.join(mnist_grid_dir, "val/annotation_coco.json"), 'w') as f:
        json.dump(val_ds, f)

def fetch_mnist_grid_img(coco_id, with_anns=True, coco_object=None):
    # path = osp.join(mnist_grid_dir, f"val/{coco_id}.jpg")
    # img = plt.imread(path)
    if coco_object is None:
        annFile = osp.join(mnist_grid_dir, "val/annotation_coco.json")
        coco_object = COCO(annFile)
    img_info = coco_object.loadImgs([int(coco_id)])[0]
    img = io.imread('%s/%s/%s'%(mnist_grid_dir,"val",img_info['file_name']), as_gray=True)
    if with_anns:
        annIds = coco_object.getAnnIds(imgIds=img_info['id'], iscrowd=None)
        anns = coco_object.loadAnns(annIds)
        return img, anns
    else:
        return img


def create_mnist_grid_coco_datapoint(base_dataset, n_digits, dp_id, ann_id,
    img_dir, category_names, scale_factor):
    base_images = base_dataset["images"]
    labels = base_dataset["labels"]
    imgs_by_label = base_dataset["images by label"]

    sublabels = []
    while 6 not in sublabels:
        indices = np.random.choice(range(len(base_images)), size=n_digits, replace=False)
        sublabels = labels[indices].numpy().tolist()
        subimgs = base_images[indices]
    
    subimgs = F.interpolate(subimgs.float().unsqueeze(1), scale_factor=(scale_factor, scale_factor),
        mode="bilinear", align_corners=False).squeeze(1).round().byte()
    superimage = [torch.cat(list(subimgs[x:x+5]), 1) for x in range(0,25,5)]
    superimage = torch.cat(superimage, 0)
    dh = int(scale_factor*28)

    def inrange(x,y):
        return x>=0 and y>=0 and x<5 and y<5

    img_height, img_width = superimage.shape
    anns = []
    relation_dict = {
        "above": (0,1),
        "left of": (1,0),
        "right of": (-1,0),
        "below": (0,-1),
    }
    for index, digit in enumerate(sublabels):
        if digit not in [6,7]:
            continue

        y,x = index // 5, index % 5
        if digit == 6:
            category = "all 6s"
            flag = True

        elif digit == 7:
            query_digit = 6
            flag = False
            for relation in relation_dict:
                dx,dy = relation_dict[relation]
                if inrange(x+dx, y+dy) and sublabels[(y+dy)*5 + x+dx] == query_digit:
                    flag = True
                    category = f"7 {relation} 6"
            if flag is False:
                category = "other 7s"

        if category in category_names:
            category_id = category_names.index(category)
            mask = torch.zeros_like(superimage)
            y_offset = y*dh
            x_offset = x*dh
            mask[y_offset:y_offset+dh, x_offset:x_offset+dh] = (subimgs[index] > 128)
            y1,x1 = mask.nonzero().min(0).values
            y2,x2 = mask.nonzero().max(0).values
            #seg = cocomask.encode(np.asfortranarray(mask.numpy()))
            seg = mask2poly(mask)
            if len(seg[0]) <= 4:
                if len(seg) > 1:
                    seg = seg[1:]
                else:
                    raise ValueError("bad mask")
            annotation = {
                "id": ann_id,
                "image_id": dp_id,
                "category_id": category_id,
                "segmentation": seg,
                "area": (subimgs[index] / 256.).sum().item(),
                "bbox": [x1.item(),y1.item(),(x2-x1).item(),(y2-y1).item()],
                "iscrowd": 0,
            }
            ann_id += 1
            anns.append(annotation)

    fn = f"{dp_id}.jpg"
    img_info = {
        "id": dp_id,
        "width": img_width,
        "height": img_height,
        "file_name": fn,
    }
    plt.imsave(osp.join(img_dir, fn), arr=superimage.numpy())
    assert len(anns) > 0, "each img should have at least 1 annotation"
    return img_info, anns