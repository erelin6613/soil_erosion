import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.cuda as cuda
from torch.utils.data import DataLoader, random_split
from pytorch_toolbelt import losses as L
import albumentations as A
from segmentation_models_pytorch.utils import metrics
import pytorch_lightning as pl
from matplotlib.pylab import plt
import random
from tqdm.notebook import tqdm

import sys
sys.path.append(os.getcwd())

from src.engine import load_model, keep_log
from src.dataset import ChipsDataset, val_tfs
from src.losses import MaskedDiceBCELoss, DiceBCELoss, FocalDiceLoss, FocalLoss, MaskedLoss, FocalLoss2
from src.poly_lr_decay import PolynomialLRDecay

with open('src/config.yaml') as f:
    config = yaml.safe_load(f)

def set_seed(seed=77):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# How do I disable determenistic for upsample_bilinear2d_backward_cuda?
# set_seed()

config['weigths'] = 'none'
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'


train_tfs = A.Compose([
    A.Transpose(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.Rotate(p=0.5),
    A.pytorch.transforms.ToTensorV2(p=1.0)])

val_tfs = A.Compose([
    A.pytorch.transforms.ToTensorV2(p=1.0)])

dataset = ChipsDataset()
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)
test_size = len(dataset) - train_size - val_size

train_set, valid_set, test_set = random_split(
    dataset, [train_size, val_size, test_size])
train_set.transforms = train_tfs
valid_set.transforms = val_tfs
test_set.transforms = val_tfs

criterion = DiceBCELoss(dice_weight=0.2, bce_weight=0.8)
metric = metrics.IoU(threshold=0.5)


class PLModel(pl.LightningModule):

    def __init__(self, model, lr):
        super(PLModel, self).__init__()
        self.model = model
        self.learning_rate = lr

        self.scores = pd.DataFrame()

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, z = batch
        y_hat = self(x)
        # print(y_hat.shape, y.shape)
        # loss = criterion(y_hat, y, z)
        loss = criterion(y_hat, y)
        score = metric(y_hat, y)
        result = {'loss': loss, 'train_iou': score}
        self.log('train_loss', loss)
        self.log('train_iou', score)
        return result

    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        # print(x.shape, y.shape)
        y_hat = self(x)
        # print(y_hat.shape, y.shape)
        # loss = criterion(y_hat, y, z)
        loss = criterion(y_hat, y)
        score = metric(y_hat, y)
        result = {'valid_loss': loss, 'val_iou': score}
        self.log('valid_loss', loss)
        self.log('valid_iou', score)
        return result

    def test_step(self, batch, batch_idx):
        x, y, z = batch
        y_hat = self(x)
        # print(y_hat.shape, y.shape)
        # loss = criterion(y_hat, y, z)
        loss = criterion(y_hat, y)
        score = metric(y_hat, y)
        result = {'test_loss': loss, 'test_iou': score}
        self.log('test_loss', loss)
        self.log('test_iou', score)
        return result

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set, self.val_set = train_set, valid_set

        if stage == 'test' or stage is None:
            self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          shuffle=True,
                          batch_size=config['batch_size'],
                          pin_memory=True,
                          num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=config['batch_size'],
                          pin_memory=True,
                          shuffle=False,
                          num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=config['batch_size'],
                          pin_memory=True,
                          shuffle=False,
                          num_workers=8)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['lr'])

        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                            config['lr'],
                                            epochs=config['epochs'],
                                            steps_per_epoch=len(train_set)//config['batch_size'])

        return (
            [optimizer],
            [{'scheduler': sched, 'interval': 'epoch', 'monitor': 'val_loss'}],
        )


def main():
    model = load_model(config['model'].lower(), path=None, device=config['device'], channels=3)
    model_version = 'nir_tci_multitemporal'

    if config['early_stopping']:
        save_path = f'models/chkpt_{config["model"]}_{config["weigths"]}_{config["epochs"]}_{model_version}_es.pt'
    else:
        save_path = f'models/{config["model"]}_{config["weigths"]}_{config["epochs"]}_{model_version}.pth'

    pl_model = PLModel(model, config['lr'])
    logger = pl.loggers.CSVLogger('logs',
                                  name='hrnet_logs')

    if config['early_stopping']:
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
           monitor='valid_loss',
           min_delta=config['es_delta'],
           patience=config['es_patience'],
           verbose=False,
           mode='min'
        )
        trainer = pl.Trainer(max_epochs=config['epochs'],
                             gpus=1 if config['device']=='cuda' else None,
                             fast_dev_run=False,
                             logger=logger,
                             weights_save_path=None,
                             callbacks=[early_stop_callback]
                            )
    else:

        trainer = pl.Trainer(max_epochs=config['epochs'],
                             gpus=1 if config['device']=='cuda' else None,
                             fast_dev_run=False,
                             #overfit_batches=1,
                             logger=logger,
                             weights_save_path=None
                            )

    if not os.path.exists(save_path):
        pl_model.setup('fit')
        train_results = trainer.fit(pl_model)
        train_results
    else:
        model = load_model(config['model'].lower(), save_path, config['device'], channels=3)
        pl_model = PLModel(model, config['lr'])

    pl_model.setup('test')
    test_results = trainer.test(pl_model)

    torch.save(pl_model.model.state_dict(), save_path)


if __name__ == '__main__':
    main()
