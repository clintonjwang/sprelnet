import os, wandb
import numpy as np
import torch
import torchvision.datasets

from sprelnet import util

ds_folder = '/data/vision/polina/scratch/clintonw/datasets'
mnist_ds_path = os.path.join(ds_folder, "mnist.bin")
multi_mnist_ds_path = os.path.join(ds_folder, 'multi-MNIST/ds_instance.bin')

labels_to_record = ["6 left of 7", "7 right of 7", "all 7s"]
label_pairs_of_interest = [(0,7),(0,0),(-1,-5)]
# pos_label_pairs=[(0,7),]
# id_label_pairs=[(0,0),]
# neg_label_pairs=[(-1,-5),]
# rand_label_pairs=[(0,1),(0,2)]

def get_mnist():
    dataset = torchvision.datasets.MNIST(ds_folder, train=True, download=True)
    train_images = dataset.train_data
    train_labels = dataset.train_labels

    dataset = torchvision.datasets.MNIST(ds_folder, train=False, download=True)
    test_images = dataset.test_data
    test_labels = dataset.test_labels

    return (train_images, train_labels), (test_images, test_labels)

def get_multi_mnist(run=None, **kwargs):
    if os.path.exists(multi_mnist_ds_path):
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
    label_names = [f"6 {r} 7" for r in relations] + [f"7 {r} 6" for r in relations] + [
            f"7 {r} 7" for r in relations] + ["all 7s", "all 6s"]
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


def create_multi_MNIST_dataset(N_train=9000, N_test=1000, save_folder="default",
        digits_per_dim=(5,5), require_non_empty_seg=False):
    if save_folder is not None:
        if save_folder == "default":
            save_folder = os.path.dirname(multi_mnist_ds_path)
        A("create or clear folder")(save_folder)

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

    for dp_ix in range(N_train + N_test):
        sublabels = []
        repeat = True
        while repeat:
            while 7 not in sublabels:
                indices = util.sample_without_replacement(range(len(images)), size=n_digits)
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
                    digit = str(A("get number in string")(label))
                    for index, letter in enumerate(labelstr):
                        if digit == letter:
                            true_index = index - index//6
                            original_seg = images[indices[true_index]] > 128
                            x_offset = (true_index // 5)*28
                            y_offset = (true_index % 5)*28
                            seg[label_ix, x_offset:x_offset+28, y_offset:y_offset+28] = original_seg
                            if dp_ix < N_train:
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
                        if dp_ix < N_train:
                            train_label_counts[label] += 1
                        else:
                            test_label_counts[label] += 1
                    #raise ValueError(f"bad label {label}")

            repeat = (require_non_empty_seg and seg.max() == 0)

        if save_folder is None:
            dp = (superimage, seg)
        else:
            img_path = os.path.join(save_folder, f"{dp_ix}_img.pt")
            seg_path = os.path.join(save_folder, f"{dp_ix}_seg.pt")
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
    if save_folder is not None:
        util.save_binary_file(data=dataset, path=multi_mnist_ds_path)
    

    return dataset

