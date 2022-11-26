import os
import yaml
from tqdm.notebook import tqdm
import segmentation_models_pytorch as smp
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import numpy as np
import pandas as pd

from .dataset import PolyDataset, val_tfs
from .models import (make_unet,
                     make_deeplabv3,
                     make_unet_plusplus,
                     make_deeplab,
                     make_hrnet_48,
                     make_hrnet_32,
                     make_hrnet_18)

try:
    with open('./code/config.yaml') as f:
        config = yaml.safe_load(f)

except Exception:
    config = {'lr': 0.0005,
              'train_size': 0.85,
              'threshold_unetplusplus': 0.25,
              'backbone': 'resnet50',
              'model': 'UnetPlusPlus',
              'weigths': 'imagenet',
              'metric_th': 0.8,
              'epochs': 200}

config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
device = config['device']
lr = config['lr']
epochs = config['epochs']
backbone = config['backbone']

save_model = False

config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
device = config['device']
lr = config['lr']
epochs = config['epochs']
backbone = config['backbone']

metrics = {'dice_loss': smp.utils.losses.DiceLoss(),
           'f_score': smp.utils.metrics.Fscore(threshold=0.0),
           'iou_score': smp.utils.metrics.IoU(threshold=0.0)}

save_model = False

models_zoo = {
    'unet': make_unet,
    'unetplusplus': make_unet_plusplus,
    'deeplab': make_deeplabv3,
    'highresolutionnet48': make_hrnet_48,
    'highresolutionnet32': make_hrnet_32,
    'highresolutionnet18': make_hrnet_18,
}


def load_model(model_name, path=None, device='cpu', channels=3, **kwargs):

    name = model_name.split('_')[0]
    model = models_zoo[name](channels=channels, **kwargs)

    if path is not None:
        model.load_state_dict(torch.load(path,
            map_location=config['device']))

    return model.to(device)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


def get_loaders(config, return_sets=False, skip_val=False):

    dataset = PolyDataset(tiles_dir='rgb_tiles',
        polygons_path='train',
        tensor_type='torch')

    train_size = int(len(dataset)*config['train_size'])

    train_set, val_set = random_split(
        dataset, [train_size, len(dataset)-train_size])

    test_set = PolyDataset(tiles_dir='rgb_tiles',
        polygons_path='test',
        tensor_type='torch',
        transforms=val_tfs)

    if return_sets:
        if not skip_val:
            return train_set, val_set, test_set
        else:
            return dataset, test_set

    if not skip_val:
        train_loader = DataLoader(train_set, config['batch_size'])
        val_loader = DataLoader(val_set, config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_set, config['batch_size'], shuffle=False)

        return train_loader, val_loader, test_loader

    else:
        train_loader = DataLoader(dataset, config['batch_size'])
        test_loader = DataLoader(test_set, config['batch_size'], shuffle=False)

        return train_loader, test_loader


def keep_log(dict_scores, file_name):
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        df = pd.DataFrame(columns=sorted(list(dict_scores.keys())))
    df = df.append(dict_scores, ignore_index=True)
    df.to_csv(file_name, index=False)


def get_metrics(y_pred, y_true, metrics=metrics, verbose=True):
    scores = {}
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = np.where(y_pred > 0, 1, 0)
    y_true = y_true.cpu().detach().numpy()
    for name, metric in metrics.items():
        score = metric(torch.from_numpy(y_pred), torch.from_numpy(y_true))
        scores[name] = score.item()

    if verbose:
        metrics_str = ' '.join([x+': '+str(y) for x, y in scores.items()])
        print(metrics_str)

    return scores, metrics_str


def train_model(config, train_loader,
                val_loader,
                criterion=nn.CrossEntropyLoss(),
                num_epochs=1,
                device=device,
                debug_mode=False):

    if config['model'] == 'unet':
        model = make_unet(
            config['backbone'], config['weigths'], 2, 'sigmoid')

    elif config['model'] == 'deeplab':
        model = make_deeplab(2, config['weigths'])
    else:
        raise Exception('This model has not been implemented yet to be trained')

    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epochs + 1):
        tr_loss = []
        val_loss = []
        print('Epoch {}/{}'.format(epoch, num_epochs))

        for sample in tqdm(train_loader):
            if config['model'] == 'deeplab':
                if sample['image'].shape[0] == 1:
                    continue
            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            y_pred = outputs if config['model'] == 'unet' else outputs['out']
            y_true = masks.squeeze(1)
            loss = criterion(y_pred.float(), y_true.long())
            loss.backward()
            tr_loss.append(loss)
            optimizer.step()
            if debug_mode:
                break

        scores, metrics_str = get_metrics(y_pred, y_true)
        scores['epoch'] = epoch
        scores['phase'] = 'train'
        scores['model'] = config['model']+'_'+config['weigths']
        keep_log(scores, 'metrics.csv')
        print(f'Train loss: {torch.mean(torch.Tensor(tr_loss))}')

        for sample in tqdm(val_loader):
            if config['model'] == 'deeplab':
                if sample['image'].shape[0]==1:
                    continue
            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)

            with torch.no_grad():
                outputs = model(inputs)
            y_pred = outputs if config['model'] == 'unet' else outputs['out']
            y_true = masks.squeeze(1)
            loss = criterion(y_pred.float(), y_true.long())
            val_loss.append(loss)
            optimizer.step()
            if debug_mode:
                break

        scores, metrics_str = get_metrics(y_pred, y_true)
        scores['epoch'] = epoch
        scores['phase'] = 'validation'
        scores['model'] = config['model']+'_'+config['weigths']
        keep_log(scores, 'metrics.csv')
        print(f'Validation loss: {torch.mean(torch.Tensor(val_loss))}')

    return model


def eval_model(model, test_loader,
               criterion=nn.CrossEntropyLoss(),
               device=device, debug_mode=False,
               config=config):

    model.to(device)
    model.eval()

    for sample in tqdm(test_loader):
        if config['model'] == 'deeplab':
            if sample['image'].shape[0] == 1:
                continue
        inputs = sample['image'].to(device)
        masks = sample['mask'].to(device)

        with torch.no_grad():
            outputs = model(inputs)
        y_pred = outputs if config['model'] == 'unet' else outputs['out']
        y_true = masks.squeeze(1)
        loss = criterion(y_pred.float(), y_true.long())
    scores, metrics_str = get_metrics(y_pred, y_true)
    scores['epoch'] = None
    scores['phase'] = 'test'
    scores['model'] = config['model']+'_'+config['weigths']
    scores['loss'] = loss.detach().cpu().item()
    keep_log(scores, 'metrics.csv')


def get_predictions(model, test_loader, debug_mode=False, config=config):

    model.to(device)
    model.eval()

    preds = []
    input_imgs = []
    for sample in tqdm(test_loader):
        inputs = sample[0].to(device)
        for batch in inputs:
            outputs = model.predict(batch)
            preds.append(outputs.cpu().detach().permute(1, 2, 0).numpy())
            input_imgs.append(batch)

    return input_imgs, preds
