import os
import dill as pickle
import numpy as np
import torch
import torchvision.datasets
import sklearn.datasets

ds_folder = '/data/vision/polina/scratch/clintonw/datasets'
mnist_ds_path = os.path.join(ds_folder, "mnist.bin")
multi_mnist_ds_path = os.path.join(ds_folder, 'multi-MNIST/ds_instance.bin')

def get_mnist(vanilla=True, refresh_dataset=False):
    if A("path exists")(mnist_ds_path) and not refresh_dataset:
        A("load dataset")(mnist_ds_path)
    else:
        dataset = torchvision.datasets.MNIST(A("get known path")("temporary folder"), train=True, download=True)
        train_images = dataset.train_data
        train_labels = dataset.train_labels

        dataset = torchvision.datasets.MNIST(A("get known path")("temporary folder"), train=False, download=True)
        test_images = dataset.test_data
        test_labels = dataset.test_labels

        if vanilla:
            return (train_images, train_labels), (test_images, test_labels)

        train_labels, test_labels = train_labels.numpy(), test_labels.numpy()
        names = ["train_%d"%ix for ix in A("range over")(train_labels)] + ["test_%d"%ix for ix in A("range over")(test_labels)]
        attributes = [{"original split":"training"}] * len(train_labels) + [{"original split":"test"}] * len(test_labels)
        images = A("instantiate images")(A("concatenate")([train_images, test_images]))
        labels = A("concatenate")([train_labels, test_labels])
        observations = [{"image":images[ix], "label":int(labels[ix])} for ix in A("range over")(images)]

        datapoints = A("run action in batches")(batch_size=10000, action="make new datapoints in parallel",
            iterated_args={"names":names, "attributes":attributes, "observations":observations})

        #image_loader = lambda dp: A("get observed value")(dp, "image")
        A("normalize all images in dataset (0-1)")()
        A("convert all images to torch tensors")()
        dataset = {"datapoints":datapoints}

        with open(mnist_ds_path, "wb") as opened_file:
            pickle.dump(dataset, opened_file)

def create_multi_MNIST_dataset(N_train=9000, N_test=1000, save_folder="default",
        digits_per_dim=(5,5), require_non_empty_seg=False):
    if save_folder is not None:
        if save_folder == "default":
            save_folder = os.path.dirname(multi_mnist_ds_path)
        A("create or clear folder")(save_folder)

    train_mnist, test_mnist = get_mnist(vanilla=True)
    images = torch.cat([train_mnist[0], test_mnist[0]], 0)
    labels = torch.cat([train_mnist[1], test_mnist[1]], 0)

    relations = ["above", "left of", "right of", "below",
        #"imm. above", "imm. left of", "imm. right of", "imm. below",
        #"adjacent to", "horiz. next to", "vert. next to",
        #"between"
    ]
    new_label_names = [f"6 {r} 7" for r in relations] + [f"7 {r} 6" for r in relations] + [
            f"7 {r} 7" for r in relations] + ["all 7s", "all 6s"]
    #"9 immediately above 7", "9 immediately left of 7",
    #"7 adjacent to 6", "3 adjacent to 4",
    n_labels = len(new_label_names)

    n_digits = np.prod(digits_per_dim)
    image_size = [d*28 for d in digits_per_dim]

    min_samples_per_label = 16
    train_label_counts = {label:0 for label in new_label_names}
    test_label_counts = {label:0 for label in new_label_names}
    relation_dict = {
        "above": {"regex": lambda m,n: f"{m}.....{n}", "span index":"start"},
        "below": {"regex": lambda m,n: f"{n}.....{m}", "span index":"end"},
        "left of": {"regex": lambda m,n: f"{m}{n}", "span index":"start"},
        "right of": {"regex": lambda m,n: f"{n}{m}", "span index":"end"},
    }

    new_datapoints = []

    for dp_ix in range(N_train + N_test):
        sublabels = []
        repeat = True
        while repeat:
            while 7 not in sublabels:
                indices = A("sample without replacement")(range(len(images)), size=n_digits)
                sublabels = labels[indices].numpy().tolist()

            for ix in range(20,0,-5):
                sublabels.insert(ix,"|")
            labelstr = "".join(map(str, sublabels))
            superimage = images[indices]
            superimage = [torch.cat(list(superimage[x:x+5]), 1) for x in range(0,25,5)]
            superimage = torch.cat(superimage, 0)
            superimage.unsqueeze_(0)

            seg = torch.zeros(n_labels,*image_size, dtype=bool)
            for label_ix, label in enumerate(new_label_names):
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
                    for match in A("get all regex matches")(d["regex"](m,n), labelstr):
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

    dataset = {"image size": image_size, "digits per image": n_digits,
        "train datapoints":new_datapoints[:N_train], "train label counts":train_label_counts,
        "test datapoints":new_datapoints[N_train:], "test label counts":test_label_counts,
    }
    if save_folder is not None:
        dataset["datapoint loader"] = lambda dp: (torch.load(dp["image path"]).cuda()/255., torch.load(dp["seg path"]).float().cuda())
        with open(multi_mnist_ds_path, "wb") as opened_file:
            pickle.dump(dataset, opened_file)

    return dataset