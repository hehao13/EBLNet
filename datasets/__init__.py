"""
Dataset setup and loaders
This file including the different datasets processing pipelines
"""

from datasets import MSD
from datasets import GDD
from datasets import Trans10k

import torchvision.transforms as standard_transforms

import transforms.joint_transforms as joint_transforms
import transforms.transforms as extended_transforms
from torch.utils.data import DataLoader


def setup_loaders(args):
    """
    Setup Data Loaders[Currently supports Cityscapes, Mapillary and ADE20kin]
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    """

    if args.dataset == 'MSD':
        args.dataset_cls = MSD
        args.train_batch_size = args.bs_mult * args.ngpu
        if args.bs_mult_val > 0:
            args.val_batch_size = args.bs_mult_val * args.ngpu
        else:
            args.val_batch_size = args.bs_mutl * args.ngpu
    elif args.dataset == 'GDD':
        args.dataset_cls = GDD
        args.train_batch_size = args.bs_mult * args.ngpu
        if args.bs_mult_val > 0:
            args.val_batch_size = args.bs_mult_val * args.ngpu
        else:
            args.val_batch_size = args.bs_mutl * args.ngpu
    elif args.dataset == 'Trans10k':
        args.dataset_cls = Trans10k
        args.train_batch_size = args.bs_mult * args.ngpu
        if args.bs_mult_val > 0:
            args.val_batch_size = args.bs_mult_val * args.ngpu
        else:
            args.val_batch_size = args.bs_mult * args.ngpu
    else:
        raise Exception('Dataset {} is not supported'.format(args.dataset))

    # Readjust batch size to mini-batch size for apex
    if args.apex:
        args.train_batch_size = args.bs_mult
        args.val_batch_size = args.bs_mult_val

    args.num_workers = 4 * args.ngpu
    if args.test_mode:
        args.num_workers = 1

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Geometric image transformations

    if args.dataset == 'GDD':
        train_joint_transform_list = [
            joint_transforms.Resize(args.crop_size),
            joint_transforms.RandomHorizontallyFlip()]
        val_joint_transform_list = [
            joint_transforms.Resize(args.crop_size)]

    if args.dataset == 'MSD':
        train_joint_transform_list = [
            joint_transforms.Resize(args.crop_size),
            joint_transforms.RandomHorizontallyFlip()]
        val_joint_transform_list = [
            joint_transforms.Resize(args.crop_size)]

    if args.dataset == 'Trans10k':
        train_joint_transform_list = [
            joint_transforms.Resize(args.crop_size)]
        val_joint_transform_list = [
            joint_transforms.Resize(args.crop_size)]

    # Image appearance transformations
    train_input_transform = []
    if args.color_aug:
        train_input_transform += [extended_transforms.ColorJitter(
            brightness=args.color_aug,
            contrast=args.color_aug,
            saturation=args.color_aug,
            hue=args.color_aug)]

    if args.bblur:
        train_input_transform += [extended_transforms.RandomBilateralBlur()]
    elif args.gblur:
        train_input_transform += [extended_transforms.RandomGaussianBlur()]
    else:
        pass

    train_input_transform += [standard_transforms.ToTensor(),
                              standard_transforms.Normalize(*mean_std)]
    train_input_transform = standard_transforms.Compose(train_input_transform)

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    target_transform = extended_transforms.MaskToTensor()


    ## relax the segmentation border
    if args.jointwtborder: 
        target_train_transform = extended_transforms.RelaxedBoundaryLossToTensor(args.dataset_cls.ignore_label, 
            args.dataset_cls.num_classes)
    else:
        target_train_transform = extended_transforms.MaskToTensor()

    edge_map = args.joint_edge_loss_light_cascade
    if args.dataset == 'MSD':
        train_set = args.dataset_cls.MSDDateset(
            'semantic', 'train', args.maxSkip,
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            target_transform=target_train_transform,
            dump_images=args.dump_augmentation_images,
            class_uniform_pct=args.class_uniform_pct,
            class_uniform_title=args.class_uniform_tile,
            test=args.test_mode,
            cv_split=args.cv,
            scf=args.scf,
            hardnm=args.hardnm,
            edge_map=edge_map,
            thicky=args.thicky
        )
        val_set = args.dataset_cls.MSDDateset(
            'semantic', 'test', 0,
            joint_transform_list=val_joint_transform_list,
            transform=val_input_transform,
            target_transform=target_transform,
            test=False,
            cv_split=args.cv,
            scf=None)
    elif args.dataset == 'GDD':
        train_set = args.dataset_cls.GDDDateset(
            'semantic', 'train', args.maxSkip,
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            target_transform=target_train_transform,
            dump_images=args.dump_augmentation_images,
            class_uniform_pct=args.class_uniform_pct,
            class_uniform_title=args.class_uniform_tile,
            test=args.test_mode,
            cv_split=args.cv,
            scf=args.scf,
            hardnm=args.hardnm,
            edge_map=edge_map,
            thicky=args.thicky
        )
        val_set = args.dataset_cls.GDDDateset(
            'semantic', 'test', 0,
            joint_transform_list=val_joint_transform_list,
            transform=val_input_transform,
            target_transform=target_transform,
            test=False,
            cv_split=args.cv,
            scf=None)
    elif args.dataset == 'Trans10k':
        train_set = args.dataset_cls.Trains10kDataset(
            'semantic', 'train', args.maxSkip,
            joint_transform_list=train_joint_transform_list,
            transform=train_input_transform,
            target_transform=target_train_transform,
            dump_images=args.dump_augmentation_images,
            class_uniform_pct=args.class_uniform_pct,
            class_uniform_title=args.class_uniform_tile,
            test=args.test_mode,
            cv_split=args.cv,
            scf=args.scf,
            hardnm=args.hardnm,
            edge_map=edge_map,
            thicky=args.thicky)
        val_set = args.dataset_cls.Trains10kDataset(
            'semantic', 'validation', 0,
            joint_transform_list=val_joint_transform_list,
            transform=val_input_transform,
            target_transform=target_transform,
            test=False,
            cv_split=args.cv,
            scf=None)
    else:
        raise Exception('Dataset {} is not supported'.format(args.dataset))
    
    if args.apex:
        from datasets.sampler import DistributedSampler
        train_sampler = DistributedSampler(train_set, pad=True, permutation=True, consecutive_sample=False)
        val_sampler = DistributedSampler(val_set, pad=False, permutation=False, consecutive_sample=False)

    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size,
                              num_workers=args.num_workers, shuffle=(train_sampler is None), drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                            num_workers=args.num_workers // 2, shuffle=False, drop_last=False, sampler=val_sampler)

    return train_loader, val_loader, train_set