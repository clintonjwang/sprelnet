import wandb
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.callback import Callback
import json
from wandb.fastai import WandbCallback
from functools import partialmethod

from sprelnet import util


def to_wandb_img(tensor):
    return wandb.Image(util.to_numpy(tensor))


# wrapper for logging masks to W&B
def wb_mask(bg_img, pred_mask=[], true_mask=[]):
    masks = {}
    if len(pred_mask) > 0:
        masks["prediction"] = {"mask_data" : pred_mask}
    if len(true_mask) > 0:
        masks["ground truth"] = {"mask_data" : true_mask}
    return wandb.Image(bg_img, classes=class_set, masks=masks)


# Custom callback for logging images to W&B
class LogImagesCallback(Callback):
    def __init__(self, learn):
        self.learn = learn

    # log semantic segmentation masks
    def on_epoch_end(self, epoch, n_epochs, **kwargs):
        # optionally limit all these to store fewer images
        # e.g. by adding [:num_log] to every line
        train_batch = self.learn.data.train_ds
        train_ids = [a.stem for a in self.learn.data.train_ds.items]
        valid_batch = self.learn.data.valid_ds
        val_ids = [a.stem for a in self.learn.data.valid_ds.items]

        train_masks = []
        valid_masks = []

        # save training and validation predictions
        # note: we're training for 1 epoch for brevity, but this approach
        # will create a new version of the artifact for each epoch
        train_res_at = wandb.Artifact("train_pred_" + wandb.run.id, "train_epoch_preds")
        val_res_at = wandb.Artifact("val_pred_" + wandb.run.id, "val_epoch_preds")
        # store all final results in a single artifact across experiments and
        # model variants to easily compare predictions
        final_model_res_at = wandb.Artifact("resnet_pred", "model_preds")


        main_columns = ["id", "prediction", "ground_truth"]
        # we'll track the IOU for each class
        main_columns.extend(["iou_" + s for s in util.BDD_CLASSES])
        # create tables
        train_table = wandb.Table(columns=main_columns)
        val_table = wandb.Table(columns=main_columns)
        model_res_table = wandb.Table(columns=main_columns)


        for batch_masks, batch, batch_ids, table, phase in zip([train_masks, valid_masks],
                                            [train_batch, valid_batch], 
                                            [train_ids, val_ids],
                                            [train_table, val_table],
                                            ["train", "val"]):
            for i, img in enumerate(batch):
                # log raw image as array
                orig_image = img[0]
                bg_image = image2np(orig_image.data*255).astype(np.uint8)

                # verify prediction from the model
                prediction = self.learn.predict(img[0])[0]
                prediction_mask = image2np(prediction.data).astype(np.uint8)

                # ground truth mask
                ground_truth = img[1]
                true_mask = image2np(ground_truth.data).astype(np.uint8)

                # score masks: what is the IOU for each class?
                per_class_scores = [util.iou_flat(prediction_mask, true_mask, i) for i in util.BDD_IDS]
                row = [str(batch_ids[i]), wb_mask(bg_image, pred_mask=prediction_mask), 
                                  wb_mask(bg_image, true_mask=true_mask)]
                row.extend(per_class_scores)
                table.add_data(*row)
                # only for last epoch
                if phase == "val" and epoch == n_epochs - 1:
                    model_res_table.add_data(*row)

        train_res_at.add(train_table, "train_epoch_results")
        val_res_at.add(val_table, "val_epoch_results")
        # by reference
        final_model_res_at.add(model_res_table, "model_results")
        wandb.run.log_artifact(train_res_at)
        wandb.run.log_artifact(val_res_at)
        wandb.run.log_artifact(final_model_res_at)