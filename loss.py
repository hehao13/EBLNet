"""
Loss.py
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg

_euler_num = 2.718281828  # euler number
_pi = 3.14159265  # pi
_ln_2_pi = 1.837877  # ln(2 * pi)
_CLIP_MIN = 1e-6  # min clip value after softmax or sigmoid operations
_CLIP_MAX = 1.0  # max clip value after softmax or sigmoid operations
_POS_ALPHA = 5e-4  # add this factor to ensure the AA^T is positive definite
_IS_SUM = 1  # sum the loss per channel


def get_loss(args):
    """
    Get the criterion based on the loss function
    args: commandline arguments
    return: criterion, criterion_val
    """

    if args.img_wt_loss:
        criterion = ImageBasedCrossEntropyLoss2d(
            classes=args.dataset_cls.num_classes, size_average=True,
            ignore_index=args.dataset_cls.ignore_label,
            upper_bound=args.wt_bound).cuda()
    elif args.joint_edge_loss_light_cascade:
        criterion = JointEdgeSegLightLossCascade(classes=args.dataset_cls.num_classes,
                                                 ignore_index=args.dataset_cls.ignore_label,
                                                 edge_weight=args.edge_weight,
                                                 body_weight=args.body_weight,
                                                 seg_weight=args.seg_weight,
                                                 ohem=args.ohem,
                                                 dice=args.dice_loss,
                                                 num_cascade=args.num_cascade).cuda()
    criterion_val = CrossEntropyLoss2d(size_average=True,
                                       weight=None,
                                       ignore_index=args.dataset_cls.ignore_label).cuda()

    return criterion, criterion_val


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, valid_mask):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        device = predict.device
        target = target.contiguous().view(target.shape[0], -1)
        target_gpu = target.clone().cuda(device=device)
        valid_mask_gpu = valid_mask.clone().cuda(device=device)
        valid_mask_gpu = valid_mask_gpu.contiguous().view(valid_mask.shape[0], -1)

        num = torch.sum(torch.mul(predict, target_gpu) * valid_mask_gpu, dim=1) * 2 + self.smooth
        den = torch.sum((predict.pow(self.p) + target_gpu.pow(self.p)) * valid_mask_gpu, dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):

        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculate_weights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculate_weights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()

            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)),
                                  targets[i].unsqueeze(0))
        return loss


class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        logging.info("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        # self.weight = weight

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class OhemCrossEntropy2dTensor(nn.Module):
    """
        Ohem Cross Entropy Tensor Version
    """

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=10000,
                 use_weight=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       weight=weight,
                                                       ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       ignore_index=ignore_index)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class JointEdgeSegLightLossCascade(nn.Module):

    def __init__(self, classes, ignore_index=255, mode='train', edge_weight=1, body_weight=1,
                 seg_weight=1, ohem=False, dice=False, num_cascade=4):
        super(JointEdgeSegLightLossCascade, self).__init__()
        self.num_classes = classes
        self.dice_loss = dice
        self.num_cascade = num_cascade
        if mode == 'train':
            if ohem:
                self.body_loss = OhemCrossEntropy2dTensor(ignore_index=ignore_index).cuda()
                self.seg_loss = OhemCrossEntropy2dTensor(ignore_index=ignore_index).cuda()
            else:
                self.body_loss = nn.CrossEntropyLoss(ignore_index=ignore_index).cuda()
                self.seg_loss = nn.CrossEntropyLoss(ignore_index=ignore_index).cuda()
        elif mode == 'val':
            self.seg_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
            self.body_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        if self.dice_loss:
            self.edge_loss = BinaryDiceLoss()

        self.ignore_index = ignore_index
        self.edge_weight = edge_weight
        self.body_weight = body_weight
        self.seg_weight = seg_weight

    def bce2d(self, input, target):
        """
        For edge
        """
        target = target.unsqueeze(1)
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight).cuda()
        log_p = log_p.cuda()
        target_t = target_t.cuda()

        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        return loss

    def forward(self, inputs, gts):
        seg_ins, body_ins, edge_ins = inputs
        mask, body_mask, edge_mask = gts

        losses = {}
        for i in range(self.num_cascade):
            seg_in, body_in, edge_in = seg_ins[i], body_ins[i], edge_ins[i]

            losses[f'seg_loss_layer{4 - i}'] = self.seg_weight * self.seg_loss(seg_in, mask.clone())
            losses[f'body_loss_layer{4 - i}'] = self.body_weight * self.body_loss(body_in, body_mask.clone())
            if not self.dice_loss:
                losses[f'edge_loss_layer{4 - i}'] = self.edge_weight * self.bce2d(edge_in, edge_mask.clone())
            else:
                device = edge_in.device
                edge_mask = edge_mask.clone()
                edge_mask.to(device)
                valid = torch.ones_like(edge_mask)
                edge_in = F.sigmoid(edge_in)
                losses[f'edge_loss_layer{4 - i}'] = self.edge_weight * self.edge_loss(edge_in, edge_mask, valid)
        return losses
