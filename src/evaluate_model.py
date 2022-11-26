import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import pickle
import yaml

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import torch
torch.manual_seed(999)
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
# default `log_dir` is "runs" - we'll be more specific here
from sklearn.metrics import mean_absolute_error

from tqdm import tqdm

import numpy as np
import glob
import os
import cv2

import sys
sys.path.append(os.path.abspath(os.getcwd()))

from model import U2NET
from model import U2NETP
from data_loader import Rescale
from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset


def normPRED(d):
    ma = np.max(d)
    mi = np.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def find_best_ckpt(ckpt_folder):
    best_score, best_path = 10e3, None

    candidates = [x for x in os.listdir(ckpt_folder) if "_" in x]
    for path in candidates:
        score = float(path.split("_")[-1].split(".")[0])
        # print(score)
        if score < best_score:
            best_score = score
            best_path = path
    return os.path.join(ckpt_folder, best_path)

def calculate_iou(pred_mask, gt_mask, threshold=0.5):
    pred_mask = np.where(pred_mask > threshold, 1, 0)
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def evaluate(config):

    checkpoint = config["evaluate"]["checkpoint_path"]
    version = config["train"]["model"]["version"]
    # checkpoint = find_best_ckpt("saved_models/u2net")
    print(f"Using checkpoint: {checkpoint}")
    # img_dir = "datasets/resized_images"
    # mask_dir = "datasets/masks"
    # os.makedirs("datasets/preds", exist_ok=True)
    img_dir = "DUTS-TE/DUTS-TE-Image"
    mask_dir = "DUTS-TE/DUTS-TE-Mask"
    os.makedirs("DUTS-TE/DUTS-TE-Pred", exist_ok=True)

    images = os.listdir(img_dir)
    images = [os.path.join(img_dir, x) for x in images]
    net = U2NET(4, 1)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    net.eval()
    tfs = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])

    masks = [x.replace("DUTS-TE-Image", "DUTS-TE-Mask") for x in images]
    # masks = [x.replace("resized_images", "masks") for x in images]
    masks = [".".join(x.split(".")[:-1]+["png"]) for x in masks]
    dataset = SalObjDataset(images, masks, transform=tfs)

    df = pd.DataFrame()
    thresholds = list(np.arange(0, 1, 0.05))
    pbar = tqdm(thresholds)
    best_thresh, best_iou = 0.5, 0

    for thresh in pbar:
        avg_mae, avg_iou = [], []
        for i, sample in enumerate(dataset):

            inputs_v = Variable(
                torch.tensor(sample["image"]), requires_grad=False)
            #print(sample["image"])
            labels_v = Variable(
                torch.tensor(sample["label"]), requires_grad=False)
            try:
                d0, d1, d2, d3, d4, d5, d6 = net(inputs_v.unsqueeze(0))
            except Exception as e:
                print("Image format error:")
                print(e)
                continue

            pred = d0[0][0] * 255
            idx = int(sample["imidx"].item())
            pred_path = dataset.label_name_list[idx]

            # pred_path = pred_path.replace("masks", "preds")
            pred_path = pred_path.replace(
                "DUTS-TE/DUTS-TE-Mask", "DUTS-TE/DUTS-TE-Pred")
            pred_path = pred_path.replace(f".{pred_path.split('.')[-1]}", ".png")
            cv2.imwrite(pred_path, pred.detach().numpy().astype(np.uint8))

            iou = calculate_iou(d0[0][0].detach().numpy(), sample["label"][0].numpy(), thresh)
            mae = mean_absolute_error(d0[0][0].detach().numpy(), sample["label"][0].numpy())
            df = df.append(
                {"image": dataset.image_name_list[i], "mae": mae, "iou": iou, "thresh": thresh}, ignore_index=True
            )
            avg_mae.append(mae)
            avg_iou.append(iou)
        pbar.set_postfix({"MAE": np.mean(avg_mae), "IoU": np.mean(avg_iou)})

        if np.mean(avg_iou) > best_iou:
            best_thresh = thresh
            best_iou = np.mean(avg_iou)

    df = df[df.thresh==best_thresh]
    # df.to_csv(f"runs/{version}_{np.mean(avg_mae)}.csv", index=False)
    df.to_csv(f"runs/{version}_{np.mean(avg_mae)}_TE.csv", index=False)
    print("Mean Absolute Error:", np.mean(avg_mae))
    print("Best IoU:", best_iou)
    print("Best thresh:", best_thresh)

    # test_image = "302150865_3396877807206671_6757778381766961437_n.jpg"
    # dataset = SalObjDataset([test_image], [test_image], transform=tfs)
    # # print
    # inputs_v = Variable(
    #     torch.tensor(dataset[0]["image"]), requires_grad=False)
    # result = net(inputs_v.unsqueeze(0))
    # result = result[0][0][0].detach().numpy()
    # print(result.shape)
    # cv2.imwrite("test.png", (255*result).astype(np.uint8))

if __name__ == '__main__':
    with open("params.yaml") as f:
        config = yaml.safe_load(f)
    evaluate(config)
