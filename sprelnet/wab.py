import os
import wandb, json
import torch
import numpy as np
# from fastai.vision import *
#from fastai.callbacks.hooks import *
# from fastai.callback import Callback
# from wandb.fastai import WandbCallback
from functools import partialmethod

from sprelnet import util, visualize
from sprelnet.data import mnist, pixels

# columns = ["image"]
# if dataset["name"] != "MNIST grid":
#     raise NotImplementedError("labels_to_record for non MNIST")
# for label in mnist.labels_to_record:
#     columns += [f"GT seg ({label})", f"predicted seg ({label})", f"IOU ({label})"]
# table = wandb.Table(columns=columns)

# img = to_wandb_img(X[0])
# row = [img]
# if dataset["name"] != "MNIST grid":
#     raise NotImplementedError("labels_to_record for non MNIST")
# for label in mnist.labels_to_record:
#     legend = mnist.get_multi_mnist_legend()
#     label_ix = legend[label]
#     gt_seg = to_wandb_img(Y_gt[0,label_ix])
#     pred_seg = to_wandb_img(Y_hat[0,label_ix])
#     iou = losses.iou(pred_seg > .5, gt_seg > .5)
#     row += [gt_seg, pred_seg, iou]
# table.add_data(*row)

# def update_metric(metric, value):
#     key = f"best {metric}"
#     if key in run.summary:
#         if metric in ["dice"]:
#             if value > run.summary[key]:
#                 run.summary[key] = value
#         elif metric in ["loss"]:
#             if value < run.summary[key]:
#                 run.summary[key] = value
#         else:
#             raise NotImplementedError(f"cannot handle metric '{metric}'")
#     else:
#         run.summary[key] = value
#     wandb.log({f"test {metric}": value})
#     run.summary.update()

def define_metrics(metrics):
    for metric in metrics:
        if "loss" in metric:
            wandb.define_metric("test loss", summary="min")
        elif "dice" in metric:
            wandb.define_metric("test dice", summary="max")
        elif "iou" in metric:
            wandb.define_metric("test iou", summary="max")
        else:
            raise NotImplementedError(metric)

def save_state(network, paths, run, model_artifact):
    torch.save(network.state_dict(), os.path.join(paths["model weights directory"], f"weights.state"))
    try:
        model_artifact.wait()
    except ValueError:
        pass
    run.log_artifact(model_artifact)

def clear_runs(runs=None, filters=None):
    if runs is None and filters is None:
        raise ValueError("clearing all WAndB runs is a dangerous action!")

    api = wandb.Api()
    if runs is not None:
        raise NotImplementedError()
    runs = api.runs(path=util.entity_project, filters=filters)
    for run in runs:
        run.delete(delete_artifacts=True)
    return runs

def clear_failed_runs(runs=None):
    return clear_runs(filters={"state":"failed"})


def log_relnet(relnet, dataset, epoch=None):
    # log kernels
    if dataset["name"] == "MNIST grid":
        legend = mnist.get_multi_mnist_legend(key="index")
        for label_pair in mnist.label_pairs_of_interest:
            mean,std = visualize.get_multiscale_likelihood_kernel_composite(relnet, *label_pair)
            wandb.log({f"mean kernel ({legend[label_pair[0]]} -> {legend[label_pair[1]]})": \
                        wandb.Image(visualize.add_colorbar_to_image(mean)),
                    f"std kernel ({legend[label_pair[0]]} -> {legend[label_pair[1]]})": \
                    wandb.Image(visualize.add_colorbar_to_image(std))}, step=epoch)
    
    elif dataset["name"] == "pixels":
        for label_pair in ((0,1), (0,2)):
            mean,std = visualize.get_multiscale_likelihood_kernel_composite(relnet, *label_pair)
            wandb.log({f"mean kernel ({label_pair[0]}-{label_pair[1]})": \
                    wandb.Image(visualize.add_colorbar_to_image(mean)),
                    f"std kernel ({label_pair[0]}-{label_pair[1]})": \
                    wandb.Image(visualize.add_colorbar_to_image(std))}, step=epoch)

    else:
        raise NotImplementedError("bad dataset")


def log_sample_outputs(X, Y_gt, Y_hat, dataset, phase, n, epoch=None):
    for i in range(n):
        log_mask_img(X[i], Y_gt[i], Y_hat[i], dataset=dataset, name=f"{phase} {i+1}", epoch=epoch)

def log_mask_img(x, y_gt, y_hat, dataset, name="mask image", label_subset=None, epoch=None, **kwargs):
    if dataset["name"] == "MNIST grid":
        class_labels = mnist.get_multi_mnist_legend(key="class labels")
    elif dataset["name"] == "pixels":
        class_labels = pixels.class_labels
    else:
        raise NotImplementedError

    wandb.log({name: to_wandb_seg(x, y_gt, class_labels=class_labels, label_subset=label_subset, 
        prediction=y_hat, **kwargs)}, step=epoch)
    # wandb.log({"true seg":to_wandb_seg(x, y_gt, class_labels),
    #     "predicted seg":to_wandb_seg(x, y_hat, class_labels)})
    # x = to_wandb_img(x)
    # y_gt = to_wandb_img(y_gt)
    # y_hat = to_wandb_img(y_hat)
    # wandb.log({"img":x, "true seg":y_gt, "predicted seg":y_hat})

def to_wandb_seg(x, seg, class_labels=None, label_subset=None, **kwargs):
    if label_subset is None:
        label_subset = list(range(len(seg)))

    if len(kwargs) == 0:
        kwargs["mask"] = seg
    else:
        kwargs["ground_truth"] = seg
    mask_kw = {}
    for key,img in kwargs.items():
        img = util.to_numpy(img)
        flatimg = np.zeros_like(img[0])
        for fill_value, channel in enumerate(label_subset):
            flatimg += (fill_value+1) * img[channel]
        mask_kw[key] = {"mask_data":flatimg, "class_labels":class_labels}

    x = util.to_numpy(x)
    return wandb.Image(x, masks=mask_kw)

def to_wandb_img(tensor):
    return wandb.Image(util.to_numpy(tensor))

# wrapper for logging masks to W&B
def wb_mask(bg_img, pred_mask=[], true_mask=[]):
    class_set = wandb.Classes([{'name': name, 'id': id} 
                               for name, id in zip(util.BDD_CLASSES, util.BDD_IDS)])

    masks = {}
    if len(pred_mask) > 0:
        masks["prediction"] = {"mask_data" : pred_mask}
    if len(true_mask) > 0:
        masks["ground truth"] = {"mask_data" : true_mask}
    return wandb.Image(bg_img, classes=class_set, masks=masks)


# Custom callback for logging images to W&B
# class LogImagesCallback(Callback):
#     def __init__(self, learn):
#         self.learn = learn

#     # log semantic segmentation masks
#     def on_epoch_end(self, epoch, n_epochs, **kwargs):
#         # optionally limit all these to store fewer images
#         # e.g. by adding [:num_log] to every line
#         train_batch = self.learn.data.train_ds
#         train_ids = [a.stem for a in self.learn.data.train_ds.items]
#         valid_batch = self.learn.data.valid_ds
#         val_ids = [a.stem for a in self.learn.data.valid_ds.items]

#         train_masks = []
#         valid_masks = []

#         # save training and validation predictions
#         # note: we're training for 1 epoch for brevity, but this approach
#         # will create a new version of the artifact for each epoch
#         train_res_at = wandb.Artifact("train_pred_" + wandb.run.id, "train_epoch_preds")
#         val_res_at = wandb.Artifact("val_pred_" + wandb.run.id, "val_epoch_preds")
#         # store all final results in a single artifact across experiments and
#         # model variants to easily compare predictions
#         final_model_res_at = wandb.Artifact("resnet_pred", "model_preds")


#         main_columns = ["id", "prediction", "ground_truth"]
#         # we'll track the IOU for each class
#         main_columns.extend(["iou_" + s for s in util.BDD_CLASSES])
#         # create tables
#         train_table = wandb.Table(columns=main_columns)
#         val_table = wandb.Table(columns=main_columns)
#         model_res_table = wandb.Table(columns=main_columns)


#         for batch_masks, batch, batch_ids, table, phase in zip([train_masks, valid_masks],
#                                             [train_batch, valid_batch], 
#                                             [train_ids, val_ids],
#                                             [train_table, val_table],
#                                             ["train", "val"]):
#             for i, img in enumerate(batch):
#                 # log raw image as array
#                 orig_image = img[0]
#                 bg_image = image2np(orig_image.data*255).astype(np.uint8)

#                 # verify prediction from the model
#                 prediction = self.learn.predict(img[0])[0]
#                 prediction_mask = image2np(prediction.data).astype(np.uint8)

#                 # ground truth mask
#                 ground_truth = img[1]
#                 true_mask = image2np(ground_truth.data).astype(np.uint8)

#                 # score masks: what is the IOU for each class?
#                 per_class_scores = [util.iou_flat(prediction_mask, true_mask, i) for i in util.BDD_IDS]
#                 row = [str(batch_ids[i]), wb_mask(bg_image, pred_mask=prediction_mask), 
#                                   wb_mask(bg_image, true_mask=true_mask)]
#                 row.extend(per_class_scores)
#                 table.add_data(*row)
#                 # only for last epoch
#                 if phase == "val" and epoch == n_epochs - 1:
#                     model_res_table.add_data(*row)

#         train_res_at.add(train_table, "train_epoch_results")
#         val_res_at.add(val_table, "val_epoch_results")
#         # by reference
#         final_model_res_at.add(model_res_table, "model_results")
#         wandb.run.log_artifact(train_res_at)
#         wandb.run.log_artifact(val_res_at)
#         wandb.run.log_artifact(final_model_res_at)