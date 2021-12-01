import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2
from torch.utils import data

import logging
from config import cfg

num_classes = 2
ignore_label = 255
root = cfg.DATASET.GDD_DIR

label2trainid = {0: 0, 255: 1}
id2cat = {0: 'background', 1: 'glass'}

palette = [0, 0, 0, 255, 255, 255]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.int8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def make_dataset(quality, mode):
    all_items = []

    assert quality == 'semantic'
    assert mode in ['train', 'test']

    img_path = osp.join(root, mode, 'images')
    mask_path = osp.join(root, mode, 'masks')

    c_items = os.listdir(img_path)
    c_items.sort()
    mask_items = [c_item.replace('.jpg', '.png') for c_item in c_items]

    for it, mask_it in zip(c_items, mask_items):
        item = (osp.join(img_path, it), osp.join(mask_path, mask_it))
        all_items.append(item)
    logging.info(f'GDD has a total of {len(all_items)} images in {mode} phase')

    logging.info(f'GDD-{mode}: {len(all_items)} images')

    return all_items


class GDDDateset(data.Dataset):

    def __init__(self, quality, mode, maxSkip=0, joint_transform_list=None,
                 transform=None, target_transform=None, dump_images=False,
                 class_uniform_pct=None, class_uniform_title=0, test=False,
                 cv_split=None, scf=None, hardnm=0, edge_map=False, thicky=8):

        self.quality = quality
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform_list = joint_transform_list
        self.transform = transform
        self.target_transform = target_transform
        self.dump_images = dump_images
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_title = class_uniform_title
        self.scf = scf
        self.hardnm = hardnm
        self.edge_map = edge_map
        self.thicky = thicky
        if cv_split:
            self.cv_split = cv_split
            assert cv_split < cfg.DATASET.CV_SPLITS
        else:
            self.cv_split = 0

        self.data_lists = make_dataset(quality, mode)
        assert len(self.data_lists), 'Found 0 images, please check the data set'

    def __getitem__(self, index):
        token = self.data_lists[index]
        img_path, mask_path = token

        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        img_name = osp.splitext(osp.basename(img_path))[0]

        mask = np.array(mask)
        mask_copy = mask.copy()

        for k, v in label2trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                img, mask = xform(img, mask)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        if self.edge_map:
            boundary = self.get_boundary(mask, thicky=self.thicky)
            body = self.get_body(mask, boundary)
            return img, mask, body, boundary, img_name

        return img, mask, img_name

    def __len__(self):
        return len(self.data_lists)

    def build_epoch(self):
        pass

    @staticmethod
    def get_boundary(mask, thicky=8):
        tmp = mask.data.numpy().astype('uint8')
        contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(tmp)
        boundary = cv2.drawContours(boundary, contour, -1, 1, thicky)
        boundary = boundary.astype(np.float)
        return boundary

    @staticmethod
    def get_body(mask, edge):
        edge_valid = edge == 1
        body = mask.clone()
        body[edge_valid] = ignore_label
        return body



