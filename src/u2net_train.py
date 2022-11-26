import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import torch
torch.manual_seed(999)
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl

# default `log_dir` is "runs" - we'll be more specific here

from tqdm import tqdm

import numpy as np
import glob
import os

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop, HorizontalFlip
from data_loader import ToTensor, RandomRotate
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from losses import muti_bce_loss_fusion
import sys

# sys.path.append(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.getcwd()))
print(sys.path)
print(os.pardir)
from model import U2NET
from model import U2NETP
import argparse
import yaml
import logging
import warnings
# import mlflow
# import dvclive
import time

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

# ------- 1. define loss function --------


# bce_loss = nn.BCEWithLogitsLoss(size_average=True)


def f1_score(y_true:torch.Tensor, y_pred:torch.Tensor) -> torch.Tensor:
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1


def train(config):
    torch.cuda.empty_cache()
    config = yaml.safe_load(open(config))
    model_name = config['train']['model']['version']
    resume = config['train']['model']['resume']
    device = config['train']['model']['device']
    checkpoint = config['train']['model']['checkpoint_path']
    save_path = config['train']['model']['save_path']
    exp_name = config['train']['other']['exp_name']

    # ------- 2. set the directory of training dataset --------
    model_name = config['train']['model']['version'] # u2netp (small model) or u2net
    port = config['train']['other']['port']
    data_dir = os.path.join(os.getcwd() + os.sep)
    tra_image_dir = config['train']['other']['images_dir']
    tra_label_dir = config['train']['other']['masks_dir']
    label_ext = '.png'

    model_dir = os.path.join(
        os.getcwd(), 'saved_models', model_name + os.sep)
    epoch_num = config['train']['HParams']['epochs']
    batch_size_train = config['train']['HParams']['batch_size']
    lr = config['train']['HParams']['lr']
    train_num = 0
    val_num = 0
    tra_img_name_list = glob.glob(data_dir + tra_image_dir + os.sep + '*')
    tra_lbl_name_list = []

    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(data_dir + tra_label_dir + os.sep + imidx + label_ext)

    logging.info("---")
    logging.info("train images: {}".format(len(tra_img_name_list)))
    logging.info("train labels: {}".format(len(tra_lbl_name_list)))
    logging.info("---")

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            HorizontalFlip(),
            ToTensorLab(flag=0)]))

    salobj_dataloader = DataLoader(
        salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=2)

    if(model_name=='u2net'):
        net = U2NET(4, 1)
    elif(model_name=='u2netp'):
        net = U2NETP(3,1)

    if resume:
        net.load_state_dict(torch.load(checkpoint))
        print("Checkpoint loaded")
    if device == "cuda":
        net.cuda()

    # ------- 4. define optimizer --------
    logging.info("---define optimizer...")
    optimizer = optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    logging.info("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 1000 # save the model every 200 iterations
    writer = SummaryWriter('runs/bg_removal')

    with mlflow.start_run(run_name=exp_name):
        writer.add_scalar('epochs', epoch_num)
        start = time.time()
        for epoch in range(0, epoch_num):
            net.train()
            pbar = tqdm(enumerate(salobj_dataloader), total=int(len(salobj_dataset)/batch_size_train))
            for i, data in pbar:
                torch.cuda.empty_cache()
                ite_num = ite_num + 1
                ite_num4val = ite_num4val + 1

                inputs, labels = data['image'], data['label']
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)

                if torch.isnan(inputs).any():
                    continue

                if device == "cuda":
                    inputs_v, labels_v = Variable(
                        inputs.cuda(), requires_grad=False), Variable(
                        labels.cuda(),requires_grad=False)
                else:
                    inputs_v, labels_v = Variable(
                        inputs, requires_grad=False), Variable(
                        labels, requires_grad=False)

                # y zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
                loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
                loss.backward()
                optimizer.step()

                # # print statistics
                running_loss += loss.data.item()
                running_tar_loss += loss2.data.item()
                pbar.set_postfix({"loss": loss.item(), "loss2": loss2.item()})
                # del temporary outputs and losss
                del d0, d1, d2, d3, d4, d5, d6, loss2, loss

                if ite_num % save_frq == 0:
                    logging.info("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                    epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                    writer.add_scalar('train loss', running_loss / ite_num4val, global_step=int((i+1)*epoch))
                    writer.add_scalar('target loss', running_tar_loss / ite_num4val, global_step=int((i+1)*epoch))
                    torch.save(net.state_dict(), model_dir + model_name+"_bce_4_channels_new_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                    running_loss = 0.0
                    running_tar_loss = 0.0
                    net.train()  # resume train
                    ite_num4val = 0

        # ------- 6. Exporting a model from PyTorch to ONNX --------

        dummy_input = torch.randn(1, 4, 320, 320, requires_grad=True, device="cuda")
        torch.onnx.export(net, dummy_input, save_path, opset_version=11)
        version = config['train']['model']['version']
        chkpt_path = f"saved_models/{version}_last.pth"
        torch.save(net.state_dict(), chkpt_path)


criterion = muti_bce_loss_fusion
# dice_fn = DiceLoss()
metric_fn = f1_score


class PLModel(pl.LightningModule):
    def __init__(self, model, config, train_set, val_set, test_set):
        super(PLModel, self).__init__()
        self.model = model
        self.config = config
        self.learning_rate = config['train']['HParams']['lr']
        self.batch_size = config['train']['HParams']['batch_size']
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.scores = pd.DataFrame()

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def on_train_epoch_end(self):
        # print(outputs)
        chkpt_path = f"saved_models/{self.config['train']['model']['version']}_{self.current_epoch}_2.pth"
        torch.save(self.model.state_dict(), chkpt_path)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        d0, d1, d2, d3, d4, d5, d6 = self(x)
        loss2, loss = muti_bce_loss_fusion(
            d0, d1, d2, d3, d4, d5, d6, y)
        # loss = criterion(d0, y)
        y_th = torch.where(d0 > 0.5, 1, 0).long()
        metric = metric_fn(y_th, y.long())
        # dice = dice_fn(y_hat, y.long())
        # loss += dice
        self.log('train_loss', loss.item(), prog_bar=True)
        self.log('train_f1', metric.item(), prog_bar=True)
        return {'loss': loss, 'train_f1': metric}

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        d0, d1, d2, d3, d4, d5, d6 = self(x)
        loss2, loss = muti_bce_loss_fusion(
            d0, d1, d2, d3, d4, d5, d6, y)
        # loss = criterion(d0, y)
        y_th = torch.where(d0 > 0.5, 1, 0).long()
        metric = metric_fn(y_th, y.long())

        self.log('val_loss', loss.item(), prog_bar=True)
        self.log('val_f1', metric.item(), prog_bar=True)
        return {'loss': loss, 'val_f1': metric}

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        d0, d1, d2, d3, d4, d5, d6 = self(x)
        loss2, loss = muti_bce_loss_fusion(
            d0, d1, d2, d3, d4, d5, d6, y)
        # loss = criterion(d0, y)
        y_th = torch.where(d0 > 0.5, 1, 0).long()
        metric = metric_fn(y_th, y.long())
        self.log('train_loss', loss.item(), prog_bar=True)
        self.log('test_f1', loss.item(), prog_bar=True)
        return {'loss': loss, 'test_f1': metric}

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          shuffle=True,
                          batch_size=self.config["train"]["HParams"]["batch_size"],
                          pin_memory=True,
                          num_workers=2
                          )

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.config["train"]["HParams"]["batch_size"],
                          pin_memory=True,
                          shuffle=False,
                          num_workers=2
                          )

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.config["train"]["HParams"]["batch_size"],
                          pin_memory=True,
                          shuffle=False,
                          num_workers=2
                          )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # print(len(self.train_set))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, self.learning_rate,
            epochs=self.config['train']['HParams']['epochs'],
            steps_per_epoch=len(self.train_set)//self.batch_size)

        return {"optimizer": optimizer} #, "sheduler": sched}


def train_test(config):
    config = yaml.safe_load(open(config))
    img_dir = config["train"]["other"]["images_dir"]
    mask_dir = config["train"]["other"]["masks_dir"]
    tra_img_name_list = list(Path(img_dir).iterdir())
    tra_lbl_name_list = tra_img_name_list.copy()
    tra_lbl_name_list = [x.with_suffix(".png") for x in tra_img_name_list]
    tra_img_name_list = [str(x) for x in tra_img_name_list]
    tra_lbl_name_list = [str(x).replace(img_dir, mask_dir) for x in tra_lbl_name_list]

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            RandomRotate(),
            HorizontalFlip(),
            ToTensorLab(flag=0)]))

    # model = U2NET(4, 1)
    model = U2NETP(3, 1)

    tr_size = int(len(salobj_dataset) * 0.90)
    v_size = int(len(salobj_dataset) * 0.05)
    test_size = len(salobj_dataset) - tr_size - v_size

    train_set, val_set, test_set = random_split(salobj_dataset, [tr_size, v_size, test_size])

    pl_model = PLModel(model, config, train_set, val_set, test_set)
    logger = pl.loggers.TensorBoardLogger(f"runs/{config['train']['model']['version']}")

    if config["train"]["model"]["resume"]:
        chkpt_path = config["train"]["model"]["checkpoint_path"]
        pl_model.model.load_state_dict(torch.load(chkpt_path))

    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=1e-3,
        patience=7,
        verbose=False,
        mode='min')

    trainer = pl.Trainer(
        max_epochs=config['train']['HParams']['epochs'],
        gpus=1 if config['train']['model']['device']=='cuda' else None,
        # fast_dev_run=True,
        logger=logger,
        weights_save_path="saved_models",
        callbacks=[early_stop_callback]
        )

    trainer.fit(model=pl_model)
    trainer.test(model=pl_model)
    version = config['train']['model']['version']
    chkpt_path = f"saved_models/{version}_last.pth"
    torch.save(pl_model.model.state_dict(), chkpt_path)
    dummy_input = torch.randn(1, 4, 320, 320, requires_grad=True, device="cuda")
    # save_path = config["train"]["model"]["save_path"]
    torch.onnx.export(pl_model.model.to("cuda"), dummy_input, config["train"]["model"]["save_path"], opset_version=11)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    train(args.config)
	# train_test(config=args.config)
