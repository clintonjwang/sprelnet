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

ds_folder = '/data/vision/polina/scratch/clintonw/datasets'
mnist_grid_dir = "/data/vision/polina/scratch/clintonw/datasets/mnist_grid"
mnist_ds_path = osp.join(ds_folder, "mnist.bin")
multi_mnist_ds_path = osp.join(ds_folder, 'mnist_grid/ds_instance.bin')

labels_to_record = ["7 left of 6", "7 right of 7", "all 6s"]
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

def get_multi_mnist_legend(key="name"):
    relations = ["above", "left of", "right of", "below",
        #"imm. above", "imm. left of", "imm. right of", "imm. below",
        #"adjacent to", "horiz. next to", "vert. next to",
        #"between"
    ]
    label_names = [f"7 {r} 6" for r in relations] + [
            f"7 {r} 7" for r in relations] + ["all 6s"]
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

def create_multi_MNIST_dataset(N_train=9000, N_test=1000, digits_per_dim=(5,5), require_relation=True,
        format_coco=False):
    save_folder = os.path.dirname(multi_mnist_ds_path)
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    train_mnist, test_mnist = get_mnist()
    images = torch.cat([train_mnist[0], test_mnist[0]], 0)
    labels = torch.cat([train_mnist[1], test_mnist[1]], 0)

    label_names = get_multi_mnist_legend().keys()
    n_labels = len(label_names)

    n_digits = np.prod(digits_per_dim)
    image_size = [d*28 for d in digits_per_dim]

    min_samples_per_label = 16
    train_label_counts = {label:0 for label in label_names}
    test_label_counts = {label:0 for label in label_names}
    relations = ["above", "left of", "right of", "below"]
    relation_dict = {
        "above": {"regex": lambda m,n: f"{m}.....{n}", "span index":"start"},
        "left of": {"regex": lambda m,n: f"{m}{n}", "span index":"start"},
        "right of": {"regex": lambda m,n: f"{n}{m}", "span index":"end"},
        "below": {"regex": lambda m,n: f"{n}.....{m}", "span index":"end"},
    }

    new_datapoints = []

    for dp_id in range(N_train + N_test):
        sublabels = []
        repeat = True
        while repeat:
            while 6 not in sublabels:
                indices = np.random.choice(range(len(images)), size=n_digits, replace=False)
                sublabels = labels[indices].numpy().tolist()

            for ix in range(20,0,-5):
                sublabels.insert(ix,"|")
            labelstr = "".join(map(str, sublabels))
            superimage = images[indices]
            superimage = [torch.cat(list(superimage[x:x+5]), 1) for x in range(0,25,5)]
            superimage = torch.cat(superimage, 0)
            superimage.unsqueeze_(0)

            seg = torch.zeros(n_labels,*image_size, dtype=bool)
            for label_ix, label in enumerate(label_names):
                if label.startswith("all"):
                    digit = str(util.get_number_in_string(label))
                    for index, letter in enumerate(labelstr):
                        if digit == letter:
                            true_index = index - index//6
                            original_seg = images[indices[true_index]] > 128
                            x_offset = (true_index // 5)*28
                            y_offset = (true_index % 5)*28
                            seg[label_ix, x_offset:x_offset+28, y_offset:y_offset+28] = original_seg
                            if dp_id < N_train:
                                train_label_counts[label] += 1
                            else:
                                test_label_counts[label] += 1

                else:
                    for relation in relations:
                        if relation in label:
                            break
                    d = relation_dict[relation]
                    m = int(label[0])
                    n = int(label[-1])
                    for match in util.get_all_regex_matches(d["regex"](m,n), labelstr):
                        if d["span index"] == "start":
                            index = match.start()
                        elif d["span index"] == "end":
                            index = match.end() - 1
                        else:
                            raise NotImplementedError
                        true_index = index - index//6
                        original_seg = images[indices[true_index]] > 128
                        x_offset = (true_index // 5)*28
                        y_offset = (true_index % 5)*28
                        seg[label_ix, x_offset:x_offset+28, y_offset:y_offset+28] = original_seg
                        if dp_id < N_train:
                            train_label_counts[label] += 1
                        else:
                            test_label_counts[label] += 1
                    #raise ValueError(f"bad label {label}")

            repeat = (require_relation and seg[:-2].max() == 0)
            sublabels = []

        # if save_folder is None:
        #     dp = (superimage, seg)
        # else:
        img_path = os.path.join(save_folder, f"{dp_id}_img.pt")
        seg_path = os.path.join(save_folder, f"{dp_id}_seg.pt")
        torch.save(superimage, img_path)
        torch.save(seg, seg_path)
        dp = {"image path": img_path, "seg path":seg_path}
        new_datapoints.append(dp)

    dataset = {"name": "MNIST grid",
        "image size": image_size, "digits per image": n_digits,
        "train datapoints":new_datapoints[:N_train], "train label counts":train_label_counts,
        "test datapoints":new_datapoints[N_train:], "test label counts":test_label_counts,
        "legend": get_multi_mnist_legend(),
        "datapoint loader": lambda dp: (torch.load(dp["image path"]).cuda()/255.,
            torch.load(dp["seg path"]).float().cuda())
    }
    # if save_folder is not None:
    util.save_binary_file(data=dataset, path=multi_mnist_ds_path)
    
    return dataset

def mask2poly(mask):
    contours, _ = cv2.findContours(mask.numpy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for object in contours:
        coords = []
        for point in object:
            coords.append(int(point[0][0]))
            coords.append(int(point[0][1]))
        polygons.append(coords)
    return polygons

def create_mnist_grid_coco_dataset(N_train=9000, N_val=1000, digits_per_dim=(5,5)):
    if digits_per_dim != (5,5):
        raise NotImplementedError
    if os.path.exists(mnist_grid_dir):
        shutil.rmtree(mnist_grid_dir)
    os.makedirs(osp.join(mnist_grid_dir, "train"))
    os.makedirs(osp.join(mnist_grid_dir, "val"))

    train_mnist, test_mnist = get_mnist()
    base_images = torch.cat([train_mnist[0], test_mnist[0]], 0)
    labels = torch.cat([train_mnist[1], test_mnist[1]], 0)
    category_names = get_multi_mnist_legend().keys()
    n_digits = np.prod(digits_per_dim)
    coco_categories = []
    for ix, label_name in enumerate(category_names):
        if label_name.startswith("all"):
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
        img_info, anns = create_mnist_grid_coco_datapoint(base_images, labels, n_digits,
            dp_id, ann_id=len(annotations), img_dir=os.path.join(mnist_grid_dir, "train"),
            category_names=category_names)
        imgs.append(img_info)
        annotations += anns
    train_ds = {
        "images": imgs,
        "annotations": annotations,
        "categories": coco_categories,
    }
    with open(os.path.join(mnist_grid_dir, "train/annotation_coco.json"), 'w') as f:
        json.dump(train_ds, f)

    imgs = []
    annotations = []
    for dp_id in range(N_train, N_val+N_train):
        img_info, anns = create_mnist_grid_coco_datapoint(base_images, labels, n_digits,
            dp_id, ann_id=len(annotations), img_dir=os.path.join(mnist_grid_dir, "val"),
            category_names=category_names)
        imgs.append(img_info)
        annotations += anns
    val_ds = {
        "images": imgs,
        "annotations": annotations,
        "categories": coco_categories,
    }
    with open(os.path.join(mnist_grid_dir, "val/annotation_coco.json"), 'w') as f:
        json.dump(val_ds, f)

def fetch_mnist_grid_img(coco_id, with_anns=False):
    # path = osp.join(mnist_grid_dir, f"val/{coco_id}.jpg")
    # img = plt.imread(path)
    annFile = os.path.join(mnist_grid_dir, "val/annotation_coco.json")
    coco = COCO(annFile)
    img_info = coco.loadImgs([int(coco_id)])[0]
    img = io.imread('%s/%s/%s'%(mnist_grid_dir,"val",img_info['file_name']), as_gray=True)
    if with_anns:
        annIds = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        return img, anns, coco
    else:
        return img
    
def create_mnist_grid_coco_datapoint(base_images, labels, n_digits, dp_id, ann_id, img_dir, category_names):
    sublabels = []
    while 6 not in sublabels:
        indices = np.random.choice(range(len(base_images)), size=n_digits, replace=False)
        sublabels = labels[indices].numpy().tolist()
    subimgs = base_images[indices]
    superimage = [torch.cat(list(subimgs[x:x+5]), 1) for x in range(0,25,5)]
    superimage = torch.cat(superimage, 0)

    def inrange(x,y):
        return x>=0 and y>=0 and x<5 and y<5

    img_height, img_width = superimage.shape
    anns = []
    for index, digit in enumerate(sublabels):
        if digit not in [6,7]:
            continue
        for category_id, category in enumerate(category_names):
            flag = False
            y,x = index // 5, index % 5
            if category.startswith("all"):
                flag = (digit == util.get_number_in_string(category))
            else:
                relation_dict = {
                    "above": (0,1),
                    "left of": (1,0),
                    "right of": (-1,0),
                    "below": (0,-1),
                }
                for relation in relation_dict:
                    if relation in category:
                        if digit == int(category[0]): # this is the right digit
                            dx,dy = relation_dict[relation]
                            query_digit = int(category[-1])
                            if inrange(x+dx, y+dy) and sublabels[(y+dy)*5 + x+dx] == query_digit:
                                flag = True
                        break

            if flag:
                mask = torch.zeros_like(superimage)
                y_offset = y*28
                x_offset = x*28
                mask[y_offset:y_offset+28, x_offset:x_offset+28] = (subimgs[index] > 128)
                y1,x1 = mask.nonzero().min(0).values
                y2,x2 = mask.nonzero().max(0).values
                #seg = cocomask.encode(np.asfortranarray(mask.numpy()))
                seg = mask2poly(mask)
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
    plt.imsave(os.path.join(img_dir, fn), arr=superimage.numpy())
    assert len(anns) > 0, "each img should have at least 1 annotation"
    return img_info, anns