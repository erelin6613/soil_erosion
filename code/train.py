import os
# import yaml
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.cuda as cuda
from torch.utils.data import DataLoader
from pytorch_toolbelt import losses as L
# import albumentations as A
import pytorch_lightning as pl
from segmentation_models_pytorch.utils import metrics

from dataset import ChipsDataset
from losses import DiceBCELoss
from models import SimpleSegNet
# from code.poly_lr_decay import PolynomialLRDecay
from settings import MAX_EPOCHS, LEARNING_RATE, BATCH_SIZE, EARLY_STOPPING, EXPERIMENT_NAME


def set_seed(seed=77):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(66)

df = pd.read_csv("../notebooks/chip_index.csv")
df["TCI_path"] = df["img_path"]
train_set = test_set = valid_set = ChipsDataset(df[df.skip == 0].reset_index())
model = SimpleSegNet(4, 1)
criterion = DiceBCELoss()
metric = metrics.IoU(threshold=0.5)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class PLModel(pl.LightningModule):
    def __init__(self, model, lr):
        super(PLModel, self).__init__()
        self.model = model
        self.learning_rate = lr
        self.scores = pd.DataFrame()

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # print(x.dtype, y_hat.dtype, y.dtype)
        # loss = criterion(y_hat, y, z)
        loss = criterion(y_hat, y)
        score = metric(y_hat, y)
        result = {'loss': loss, 'train_iou': score}
        self.log('train_loss', loss)
        self.log('train_iou', score)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = criterion(y_hat, y, z)
        loss = criterion(y_hat, y)
        score = metric(y_hat, y)
        result = {'valid_loss': loss, 'val_iou': score}
        self.log('valid_loss', loss)
        self.log('valid_iou', score)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
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
                          batch_size=BATCH_SIZE,
                          pin_memory=True,
                          num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=BATCH_SIZE,
                          pin_memory=True,
                          shuffle=False,
                          num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=BATCH_SIZE,
                          pin_memory=True,
                          shuffle=False,
                          num_workers=8)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)

        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                            LEARNING_RATE,
                                            epochs=MAX_EPOCHS,
                                            steps_per_epoch=len(train_set)//BATCH_SIZE)

        return (
            [optimizer],
            [{'scheduler': sched, 'interval': 'epoch', 'monitor': 'val_loss'}],
        )


pl_model = PLModel(model, LEARNING_RATE)
logger = pl.loggers.CSVLogger('logs',
                              name='hrnet_logs')

# if EARLY_STOPPING:
#     early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
#        monitor='valid_loss',
#        min_delta=config['es_delta'],
#        patience=config['es_patience'],
#        verbose=False,
#        mode='min'
#     )
#     trainer = pl.Trainer(max_epochs=MAX_EPOCHS,
#                          gpus=1 if device == 'cuda' else None,
#                          fast_dev_run=False,
#                          logger=logger,
#                          weights_save_path=None,
#                          callbacks=[early_stop_callback]
#                         )
# else:

trainer = pl.Trainer(max_epochs=MAX_EPOCHS,
                     gpus=1 if device == 'cuda' else None,
                     fast_dev_run=False,
                     #overfit_batches=1,
                     logger=logger,
                     weights_save_path=None
                    )

save_path = f'logs/{EXPERIMENT_NAME}.pth'

# if not os.path.exists("experiments")
# os.makedirs("experiments", exist_ok=True)


pl_model.setup('fit')
train_results = trainer.fit(pl_model)

pl_model.setup('test')
test_results = trainer.test(pl_model)

torch.save(pl_model.model.state_dict(), save_path)
