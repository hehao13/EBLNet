"""
Miscellanous Functions
"""

import sys
import re
import os
import shutil
import time
import torch
from datetime import datetime
import logging
from subprocess import call
import shlex
from tensorboardX import SummaryWriter
import numpy as np
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
import pdb

# Create unique output dir name based on non-default command line args
def make_exp_name(args, parser):
    exp_name = '{}-{}'.format(args.dataset[:4], args.arch[:])
    dict_args = vars(args)

    # sort so that we get a consistent directory name
    argnames = sorted(dict_args)
    ignorelist = ['exp', 'arch','prev_best_filepath', 'lr_schedule', 'max_cu_epoch', 'max_epoch',
                  'strict_bdr_cls', 'world_size', 'tb_path','best_record', 'test_mode', 'ckpt'
                  'jointwtborder','syncbn'
                  ]
    # build experiment name with non-default args
    for argname in argnames:
        if dict_args[argname] != parser.get_default(argname):
            if argname in ignorelist:
                continue
            if argname == 'snapshot':
                arg_str = 'PT'
                argname = ''
            elif argname == 'nosave':
                arg_str = ''
                argname=''
            elif argname == 'freeze_trunk':
                argname = ''
                arg_str = 'ft'
            elif argname == 'syncbn':
                argname = ''
                arg_str = 'sbn'
            elif argname == 'jointwtborder':
                argname = ''
                arg_str = 'rlx_loss'
            elif argname == 'num_cascade':
                argname = ''
                arg_str = 'n_casc'
            elif argname == 'joint_edge_loss_light_cascade':
                argname = ''
                arg_str = 'jellc'
            elif argname == 'class_uniform_pct':
                argname = ''
                arg_str = 'cup'
            elif argname == 'num_points':
                argname = ''
                arg_str = 'np'
            elif argname == 'edge_weight':
                argname = ''
                arg_str = 'ew'
            elif argname == 'color_aug':
                argname = ''
                arg_str = 'ca'
            elif argname == 'crop_size':
                argname = ''
                arg_str = 'cs'
            elif isinstance(dict_args[argname], bool):
                arg_str = 'T' if dict_args[argname] else 'F'
            else:
                arg_str = str(dict_args[argname])[:7]
            if argname is not '':
                exp_name += '_{}_{}'.format(str(argname), arg_str)
            else:
                exp_name += '_{}'.format(arg_str)
    # clean special chars out    exp_name = re.sub(r'[^A-Za-z0-9_\-]+', '', exp_name)
    return exp_name

def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def save_log(prefix, output_dir, date_str, rank=0):
    fmt = '%(asctime)s.%(msecs)03d %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    filename = os.path.join(output_dir, prefix + '_' + date_str +'_rank_' + str(rank) +'.log')
    print("Logging :", filename)
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=date_fmt,
                        filename=filename, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    console.setFormatter(formatter)
    if rank == 0:
        logging.getLogger('').addHandler(console)
    else:
        fh = logging.FileHandler(filename)
        logging.getLogger('').addHandler(fh)



def prep_experiment(args, parser):
    """
    Make output directories, setup logging, Tensorboard, snapshot code.
    """
    ckpt_path = args.ckpt
    tb_path = args.tb_path
    exp_name = make_exp_name(args, parser)
    args.exp_path = os.path.join(ckpt_path, args.exp, exp_name)
    args.tb_exp_path = os.path.join(tb_path, args.exp, exp_name)
    args.ngpu = torch.cuda.device_count()
    args.date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                        'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    args.last_record = {}
    if args.local_rank == 0:
        os.makedirs(args.exp_path, exist_ok=True)
        os.makedirs(args.tb_exp_path, exist_ok=True)
        save_log('log', args.exp_path, args.date_str, rank=args.local_rank)
        open(os.path.join(args.exp_path, args.date_str + '.txt'), 'w').write(
            str(args) + '\n\n')
        writer = SummaryWriter(log_dir=args.tb_exp_path, comment=args.tb_tag)
        return writer
    return None

def evaluate_eval_for_inference(hist, dataset=None, with_f1=False, beta=1):
    """
    Modified IOU mechanism for on-the-fly IOU calculations ( prevents memory overflow for
    large dataset) Only applies to eval/eval.py
    """
    # axis 0: gt, axis 1: prediction
    all_acc = np.diag(hist).sum() / hist.sum()
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    macc = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    if with_f1:
        f1s = print_evaluate_results_f1(hist, iu, beta=beta, dataset=dataset)
    else:
        print_evaluate_results(hist, iu, dataset=dataset)
    freq = hist.sum(axis=1) / hist.sum()
    mean_iu = np.nanmean(iu)
    if with_f1:
        mean_f1 = np.nanmean(f1s)
    logging.info('mean iou {}'.format(mean_iu))
    if with_f1:
        logging.info(f'mean f1 {mean_f1}')
    logging.info(f'all acc: {all_acc}')
    logging.info(f'macc:{macc}')
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def evaluate_eval_for_inference_with_mae_ber(hist, total_maes, total_bers, total_bers_count, dataset=None):

    all_acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    macc = np.nanmean(acc_cls)
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    ber = 1.0 * total_bers / total_bers_count

    print_evaluate_results_acc_iou_mae_ber(hist, iou, acc_cls, ber, dataset=dataset)
    miou = np.nanmean(iou)
    mae = np.nanmean(np.array(total_maes))
    mBer = np.nanmean(ber)
    logging.info(f'mean iou: {miou}')
    logging.info(f'macc: {macc}')
    logging.info(f'all acc: {all_acc}')
    logging.info(f'mae: {mae}')
    logging.info(f'mBer: {mBer}')
    return all_acc, acc_cls, macc, miou, mae, mBer

def evaluate_eval(args, net, optimizer, val_loss, hist, dump_images, writer, epoch=0, dataset=None):
    """
    Modified IOU mechanism for on-the-fly IOU calculations ( prevents memory overflow for
    large dataset) Only applies to eval/eval.py
    """
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    print_evaluate_results(hist, iu,  dataset)
    freq = hist.sum(axis=1) / hist.sum()
    mean_iu = np.nanmean(iu)
    logging.info('mean_iou {}'.format(mean_iu))
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    # update latest snapshot
    if 'mean_iu' in args.last_record:
        last_snapshot = 'last_epoch_{}_mean-iu_{:.5f}.pth'.format(
            args.last_record['epoch'], args.last_record['mean_iu'])
        last_snapshot = os.path.join(args.exp_path, last_snapshot)
        try:
            os.remove(last_snapshot)
        except OSError:
            pass
    last_snapshot = 'last_epoch_{}_mean-iu_{:.5f}.pth'.format(epoch, mean_iu)
    last_snapshot = os.path.join(args.exp_path, last_snapshot)
    args.last_record['mean_iu'] = mean_iu
    args.last_record['epoch'] = epoch
    
    torch.cuda.synchronize()
    
    torch.save({
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'mean_iu': mean_iu,
        'command': ' '.join(sys.argv[1:])
    }, last_snapshot)

    # update best snapshot
    if mean_iu > args.best_record['mean_iu']:
        # remove old best snapshot
        if args.best_record['epoch'] != -1:
            best_snapshot = 'best_epoch_{}_mean-iu_{:.5f}.pth'.format(
                args.best_record['epoch'], args.best_record['mean_iu'])
            best_snapshot = os.path.join(args.exp_path, best_snapshot)
            assert os.path.exists(best_snapshot), \
                'cant find old snapshot {}'.format(best_snapshot)
            os.remove(best_snapshot)

        
        # save new best
        args.best_record['val_loss'] = val_loss.avg
        args.best_record['epoch'] = epoch
        args.best_record['acc'] = acc
        args.best_record['acc_cls'] = acc_cls
        args.best_record['mean_iu'] = mean_iu
        args.best_record['fwavacc'] = fwavacc

        best_snapshot = 'best_epoch_{}_mean-iu_{:.5f}.pth'.format(
            args.best_record['epoch'], args.best_record['mean_iu'])
        best_snapshot = os.path.join(args.exp_path, best_snapshot)
        shutil.copyfile(last_snapshot, best_snapshot)
        
    
        to_save_dir = os.path.join(args.exp_path, 'best_images')
        os.makedirs(to_save_dir, exist_ok=True)

    logging.info('-' * 107)
    fmt_str = '[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], ' +\
              '[mean_iu %.5f], [fwavacc %.5f]'
    logging.info(fmt_str % (epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))
    fmt_str = 'best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], ' +\
              '[mean_iu %.5f], [fwavacc %.5f], [epoch %d], '
    logging.info(fmt_str % (args.best_record['val_loss'], args.best_record['acc'],
                            args.best_record['acc_cls'], args.best_record['mean_iu'],
                            args.best_record['fwavacc'], args.best_record['epoch']))
    logging.info('-' * 107)

    # tensorboard logging of validation phase metrics

    writer.add_scalar('training/acc', acc, epoch)
    writer.add_scalar('training/acc_cls', acc_cls, epoch)
    writer.add_scalar('training/mean_iu', mean_iu, epoch)
    writer.add_scalar('training/val_loss', val_loss.avg, epoch)


def evaluate_eval_iou_acc_mae_ber(args, net, optimizer, val_loss, hist, mae, bers,
                                  bers_count, dump_images, writer, epoch=0, dataset=None):

    # axis 0: gt, axis 1: prediction
    acc = np.drag(hist).sum() / hist.sum()
    acc_c = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_c)
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mae = np.nanmean(np.array((mae)))
    Ber = 1.0 * bers / bers_count
    mBer = np.nanmean(Ber)

    print_evaluate_results_acc_iou_mae_ber(hist, iou, acc_c, Ber, dataset)

    mean_iou = np.nanmean(iou)
    logging.info(f'mean iou: {mean_iou}')
    logging.info(f'macc: {acc_cls}')
    logging.info(f'mae: {mae}')
    logging.info(f'mBer: {mBer}')

    if 'mean_iou' in args.last_record:
        last_snapshot = 'last_epoch_{}_mean-iou_{:.5f}.pth'.format(
            args.last_record['epoch'], args.last_record['mean_iou'])
        last_snapshot = os.path.join(args.exp_path, last_snapshot)
        try:
            os.remove(last_snapshot)
        except OSError:
            pass
    last_snapshot = 'last_epoch_{}_mean-iou_{:.5f}.pth'.format(epoch, mean_iou)
    last_snapshot = os.path.join(args.exp_path, last_snapshot)
    args.last_record['mean_iou'] = mean_iou
    args.last_record['epoch'] = epoch

    torch.cuda.synchronize()

    torch.save({
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'mean_iou': mean_iou,
        'command': ' '.join(sys.argv[1:])
    }, last_snapshot)

    if mean_iou > args.best_record['mean_iou']:
        if mean_iou > args.best_record['epoch'] != -1:
            best_snapshot = 'best_epoch_{}_mean-iou_{:5f}.pth'.format(
                args.best_record['epoch'], args.best_record['mean_iou'])
            best_snapshot = os.path.join(args.exp_path, best_snapshot)
            assert os.path.exists(best_snapshot), \
                'Cannot find old snapshot {}'.format(best_snapshot)
            os.remove(best_snapshot)

        args.best_record['val_loss'] = val_loss.avg
        args.best_record['epoch'] = epoch
        args.best_record['acc'] = acc
        args.best_record['macc'] = acc_cls
        args.best_record['mean_iou'] = mean_iou
        args.best_record['mae'] = mae
        args.best_record['mBer'] = mBer

        best_snapshot = 'best_epoch_{}_mean-iou_{:.5f}.pth'.format(
            args.best_record['epoch'], args.best_record['mean_iou'])
        best_snapshot = os.path.join(args.exp_path, best_snapshot)
        shutil.copyfile(last_snapshot, best_snapshot)

        to_save_dir = os.path.join(args.exp_path, 'best_images')
        os.makedirs(to_save_dir, exist_ok=True)
        val_visual = []

        visualize = standard_transforms.Compose([
            standard_transforms.Scale(384),
            standard_transforms.ToTensor()
        ])
        for bs_idx, bs_data in enumerate(dump_images):
            for local_idx, data in enumerate(zip(bs_data[0], bs_data[1], bs_data[2])):
                gt_pil = args.dataset_cls.colorize_mask(data[0].cpu().numpy())
                predictions_pil = args.dataset_cls.colorize_mask(data[1].cpu().numpy())
                img_name = data[2]

                prediction_fn = '{}_prediction.png'.format(img_name)
                predictions_pil.save(os.path.join(to_save_dir, prediction_fn))
                gt_fn = '{}_gt.png'.format(img_name)
                gt_pil.save(os.path.join(to_save_dir, gt_fn))
                val_visual.extend([visualize(gt_pil.convert('RGB')),
                                   visualize(predictions_pil.convert('RGB'))])
                if local_idx >= 9:
                    break
        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=10, padding=5)
        writer.add_image('imgs', val_visual, epoch)

    logging.info('-' * 107)
    fmt_str = '[epoch %d], [val loss %.5f], [acc %.5d], [macc %.5f], ' + \
              '[mean_iou %.5f], [mae %.5f], [mBer %.5f]'
    logging.info(fmt_str % (epoch, val_loss.avg, acc, acc_cls, mean_iou,
                                mae, mBer))
    fmt_str = 'best_record: [val loss %.5f], [acc %.5d], [macc %.5f], ' + \
              '[mean_iou %.5f], [mae %.5f], [mBer %.5f], [epoch %d]'
    logging.info(fmt_str % (args.best_record['val_loss'], args.best_record['acc'],
                            args.best_record['macc'], args.best_record['mean_iou'],
                            args.best_record['mae'], args.best_record['mBer'],
                            args.best_record['epoch']))
    logging.info('-' * 107)

    writer.add_scalar('training/acc', acc, epoch)
    writer.add_scalar('training/macc', acc_cls, epoch)
    writer.add_scalar('training/mean_iou', mean_iou, epoch)
    writer.add_scalar('training/val_loss', val_loss.avg, epoch)

def print_evaluate_results(hist, iu, dataset=None):
    # fixme: Need to refactor this dict
    try:
        id2cat = dataset.id2cat
    except:
        id2cat = {i: i for i in range(dataset.num_classes)}
    iu_false_positive = hist.sum(axis=1) - np.diag(hist)
    iu_false_negative = hist.sum(axis=0) - np.diag(hist)
    iu_true_positive = np.diag(hist)

    logging.info('IoU:')
    logging.info('label_id      label    iU    Precision Recall TP     FP    FN')
    for idx, i in enumerate(iu):
        # Format all of the strings:
        idx_string = "{:2d}".format(idx)
        class_name = "{:>13}".format(id2cat[idx]) if idx in id2cat else ''
        iu_string = '{:5.2f}'.format(i * 100)
        total_pixels = hist.sum()
        tp = '{:5.2f}'.format(100 * iu_true_positive[idx] / total_pixels)
        fp = '{:5.2f}'.format(
            iu_false_positive[idx] / iu_true_positive[idx])
        fn = '{:5.2f}'.format(iu_false_negative[idx] / iu_true_positive[idx])
        precision = '{:5.2f}'.format(
            iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_positive[idx]))
        recall = '{:5.2f}'.format(
            iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_negative[idx]))
        logging.info('{}    {}   {}  {}     {}  {}   {}   {}'.format(
            idx_string, class_name, iu_string, precision, recall, tp, fp, fn))

def print_evaluate_results_f1(hist, iu, beta=1, dataset=None):
    # fixme: Need to refactor this dict
    try:
        id2cat = dataset.id2cat
    except:
        id2cat = {i: i for i in range(dataset.num_classes)}
    iu_false_positive = hist.sum(axis=1) - np.diag(hist)
    iu_false_negative = hist.sum(axis=0) - np.diag(hist)
    iu_true_positive = np.diag(hist)
    f1s = []

    logging.info('IoU:')
    logging.info('label_id      label    iU    Precision Recall  TP     FP    FN    F1')
    for idx, i in enumerate(iu):
        # Format all of the strings:
        idx_string = "{:2d}".format(idx)
        class_name = "{:>13}".format(id2cat[idx]) if idx in id2cat else ''
        iu_string = '{:5.5f}'.format(i * 100)
        total_pixels = hist.sum()
        tp = '{:5.5f}'.format(100 * iu_true_positive[idx] / total_pixels)
        fp = '{:5.5f}'.format(
            iu_false_positive[idx] / iu_true_positive[idx])
        fn = '{:5.5f}'.format(iu_false_negative[idx] / iu_true_positive[idx])
        pre = iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_positive[idx])
        rec = iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_negative[idx])
        f1 = (1 + beta) * pre * rec / (beta * pre + rec)
        f1s.append(f1)
        precision = '{:5.5f}'.format(pre)
        recall = '{:5.5f}'.format(rec)
        f1 = '{:5.5f}'.format(
            (1+beta)*pre*rec/(beta*pre+rec))
        logging.info('{}    {}   {}  {}     {}  {}   {}   {}  {}'.format(
            idx_string, class_name, iu_string, precision, recall, tp, fp, fn, f1))
    return f1s

def print_evaluate_results_acc_iou_mae_ber(hist, iou, acc, Ber, dataset=None):
    try:
        id2cat = dataset.id2cat
    except:
        id2cat = {i:i for i in range(dataset.num_classes)}
    false_positive = hist.sum(axis=1) - np.diag(hist)
    false_negative = hist.sum(axis=0) - np.diag(hist)
    true_positive = np.diag(hist)

    logging.info('label_id     label     iou    acc    Ber    Precision Recall TP    FP    FN')
    for idx, i in enumerate(iou):
        idx_string = "{:2d}".format(idx)
        class_name = "{:>13}".format(id2cat[idx] if idx in id2cat else '')
        iou_string = "{:5.5f}".format(i * 100)
        total_pixels = hist.sum()
        tp = '{:5.5f}'.format(100 * true_positive[idx] / total_pixels)
        fp = '{:5.5f}'.format(false_positive[idx] / true_positive[idx])
        fn = '{:5.5f}'.format(false_negative[idx] / (true_positive[idx] + false_positive[idx]))
        precision = '{:5.5f}'.format(true_positive[idx] / (true_positive[idx] + false_positive[idx]))
        recall = '{:5.5f}'.format(true_positive[idx] / (true_positive[idx] + false_negative[idx]))
        acc_string = '{:5.5f}'.format(acc[idx])
        ber_string = '{:5.5f}'.format(Ber[idx])
        logging.info('{}    {}    {}    {}    {}    {}    {}    {}  {}    {}'.format(
            idx_string, class_name, iou_string, acc_string, ber_string,
            precision, recall, tp, fp, fn))

def cal_mae(pred, label):

    mae = np.abs(pred-label)
    mae = np.mean(mae)
    return mae

def cal_ber(pred, label, num_classes):
    bers = np.zeros((num_classes, ), dtype=np.float)
    bers_count = np.zeros((num_classes, ), dtype=np.float)
    bers_count[0] = 1

    for cls_index in range(num_classes):
        valid = label == cls_index
        if valid.sum() == 0:
            continue
        num_positive = float(np.sum(label == cls_index))
        num_negative = float(np.sum(label != cls_index))
        true_psitive = float(np.sum((pred == label) * valid))
        true_negative = float(np.sum((pred == label) * (1 - valid)))
        ber = 1 - 1/2 * (true_psitive / num_positive + true_negative / num_negative)
        bers[cls_index] = ber
        bers_count[cls_index] = 1.0

    return bers, bers_count


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

