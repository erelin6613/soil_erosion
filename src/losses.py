from pytorch_toolbelt import losses as L
import segmentation_models_pytorch as smp
import torch
from torch import log
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.modules.loss._Loss):

    def forward(self, y_pred, y_true):
        return 1 - L.functional.soft_dice_score(y_pred, y_true)


class DiceBCELoss(smp.utils.base.Loss):
    def __init__(self, dice_weight=0.8, bce_weight=0.8):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, y_pred, y_true):
        y_true = y_true.view(y_pred.shape[0], -1, y_pred.shape[-2], y_pred.shape[-1])
        # print(y_pred.max(), y_true.max())
        loss = L.JointLoss(DiceLoss(), nn.BCELoss(), self.dice_weight, self.bce_weight)

        return loss(y_pred, y_true)


class MaskedDiceBCELoss(smp.utils.base.Loss):
    def __init__(self, dice_weight=0.8, bce_weight=0.8):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, y_pred, y_true, ignore_mask):
        # print(y_pred.shape, y_true.shape, ignore_mask.shape)
        y_true_flatten = y_true.flatten()
        y_pred_flatten = y_pred.flatten()
        ignore_mask_flatten = ignore_mask.flatten()
        y_true_masked = y_true_flatten[~ignore_mask_flatten]
        y_pred_masked = y_pred_flatten[~ignore_mask_flatten]
        loss = L.JointLoss(DiceLoss(), nn.BCELoss(), self.dice_weight, self.bce_weight)
        return loss(y_pred_masked, y_true_masked)


class MaskedLoss(smp.utils.base.Loss):
    def __init__(self,
                 base_loss=L.BinaryFocalLoss(alpha=0.1, gamma=4),
                 aux_loss=DiceLoss(),
                 base_weigth=0.5,
                 aux_weigth=0.5):

        super().__init__()
        self.base_loss = base_loss
        self.aux_loss = aux_loss
        self.base_weigth = base_weigth
        self.aux_weigth = aux_weigth

    def forward(self, y_pred, y_true, ignore_mask):
        y_true_flatten = y_true.flatten()
        y_pred_flatten = y_pred.flatten()
        ignore_mask_flatten = ignore_mask.flatten()
        y_true_masked = y_true_flatten[~ignore_mask_flatten]
        y_pred_masked = y_pred_flatten[~ignore_mask_flatten]
        loss = L.JointLoss(
            self.base_loss, self.aux_loss, self.base_weigth, self.aux_weigth)
        return loss(y_pred_masked, y_true_masked)


class FocalLoss2(nn.Module):
    """
    Weighs the contribution of each sample to the loss based in the classification error.
    If a sample is already classified correctly by the CNN, its contribution to the loss decreases.
    :eps: Focusing parameter. eps=0 is equivalent to BCE_loss
    """
    def __init__(self, l=0.5, eps=1e-6):
        super(FocalLoss2, self).__init__()
        self.l = l
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.view(-1)
        # probs = torch.sigmoid(logits).view(-1)
        probs = logits.view(-1)

        losses = -(targets * torch.pow((1. - probs), self.l) * torch.log(probs + self.eps) + \
                   (1. - targets) * torch.pow(probs, self.l) * torch.log(1. - probs + self.eps))
        loss = torch.mean(losses)

        return loss

class FocalDiceLoss(smp.utils.base.Loss):
    # Idea paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9180275
    def __init__(self,
                 beta=4,
                 alpha=0.2,
                 gamma=2.5):

        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = L.functional.soft_dice_score #DiceLoss()

    def forward(self, y_pred, y_true, ignore_mask=None):
        y_true_flatten = y_true.flatten()
        y_pred_flatten = y_pred.flatten()
        if ignore_mask is not None:
            ignore_mask_flatten = ignore_mask.flatten()
            y_true_flatten = y_true_flatten[~ignore_mask_flatten]
            y_pred_flatten = y_pred_flatten[~ignore_mask_flatten]
        focal = self.focal(y_pred_flatten.reshape(1, -1),
                           y_true_flatten.reshape(1, -1))
        dice = self.dice(y_pred_flatten, y_true_flatten)
        return (self.beta * focal) - torch.log(dice)


def _neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss
