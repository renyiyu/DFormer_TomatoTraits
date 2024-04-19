import cv2
import torch
import numpy as np
from torch.utils import data
import random
import albumentations as A
# from config import config
# from train import config
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize


def tomato_train_preprocess(rgb, x):
    trans = A.Compose([
        A.LongestMaxSize(max_size=640, interpolation=1),
        A.PadIfNeeded(min_height=640, min_width=640, border_mode=0, value=(0, 0, 0)),
        ])
    rgb = trans(image=rgb)['image']
    x = trans(image=x)['image']

    # data augmentation
    if random.random() >= 0.5:
        if random.random() >= 0.5:
            rgb = cv2.flip(rgb, 1)
            x = cv2.flip(x, 1)
        if random.random() >= 0.5:
            rgb = cv2.flip(rgb, 0)
            x = cv2.flip(x, 0)
        if random.random() >= 0.5:
            rgb = cv2.flip(rgb, -1)
            x = cv2.flip(x, -1)

    if random.random() >= 0.5:
        if random.random() >= 0.5:
            rgb = cv2.rotate(rgb, 0)
            x = cv2.rotate(x, 0)
        if random.random() >= 0.5:
            rgb = cv2.rotate(rgb, 2)
            x = cv2.rotate(x, 2)

    if random.random() >= 0.5:
        trans = A.RandomBrightnessContrast()
        rgb = trans(image=rgb)['image']

    return rgb, x


def random_mirror(rgb, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        modal_x = cv2.flip(modal_x, 1)

    return rgb, modal_x


def random_scale(rgb, modal_x, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return rgb, modal_x, scale


class TrainPre(object):
    def __init__(self, rgb_norm_mean, rgb_norm_std, x_norm_mean, x_norm_std, sign=False, config=None):
        self.config = config
        self.rgb_norm_mean = rgb_norm_mean
        self.rgb_norm_std = rgb_norm_std
        self.x_norm_mean = x_norm_mean
        self.x_norm_std = x_norm_std
        self.sign = sign

    def __call__(self, rgb, modal_x):
        # data augmentation
        # rgb, modal_x = random_mirror(rgb, modal_x)

        # albumentation
        rgb, modal_x = tomato_train_preprocess(rgb, modal_x)

        # if self.config.train_scale_array is not None:
        #     rgb, modal_x, scale = random_scale(rgb, modal_x, self.config.train_scale_array)

        rgb = normalize(rgb, self.rgb_norm_mean, self.rgb_norm_std)
        modal_x = normalize(modal_x, self.x_norm_mean, self.x_norm_std)
        # if self.sign:
        #     modal_x = normalize(modal_x, [0.48,0.48,0.48], [0.28,0.28,0.28])#[0.5,0.5,0.5]
        # else:
        #     modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        # crop_size = (self.config.image_height, self.config.image_width)
        # crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        # p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        # p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        # p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)
        p_rgb = rgb
        p_modal_x = modal_x

        p_rgb = p_rgb.transpose(2, 0, 1)
        p_modal_x = p_modal_x.transpose(2, 0, 1)

        return p_rgb, p_modal_x


class ValPre(object):
    def __init__(self, rgb_norm_mean, rgb_norm_std, x_norm_mean, x_norm_std, sign=False, config=None):
        self.config = config
        self.rgb_norm_mean = rgb_norm_mean
        self.rgb_norm_std = rgb_norm_std
        self.x_norm_mean = x_norm_mean
        self.x_norm_std = x_norm_std
        self.sign = sign

    def __call__(self, rgb, modal_x):
        val_trans = A.Compose([
            A.LongestMaxSize(max_size=640, interpolation=1),
            A.PadIfNeeded(min_height=640, min_width=640, border_mode=0, value=(0, 0, 0)),
        ])
        rgb = val_trans(image=rgb)['image']
        modal_x = val_trans(image=modal_x)['image']
        rgb = normalize(rgb, self.rgb_norm_mean, self.rgb_norm_std)
        modal_x = normalize(modal_x, self.x_norm_mean, self.x_norm_std)
        # modal_x = normalize(modal_x, [0.48,0.48,0.48], [0.28,0.28,0.28])
        return rgb.transpose(2, 0, 1), modal_x.transpose(2, 0, 1)


def get_train_loader(engine, dataset, config):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root': config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    train_preprocess = TrainPre(config.rgb_norm_mean,
                                config.rgb_norm_std,
                                config.x_norm_mean,
                                config.x_norm_std,
                                config.x_is_single_channel,
                                config)

    train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler


def get_val_loader(engine, dataset, config, gpus):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root': config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    val_preprocess = ValPre(config.rgb_norm_mean,
                            config.rgb_norm_std,
                            config.x_norm_mean,
                            config.x_norm_std,
                            config.x_is_single_channel,
                            config)

    val_dataset = dataset(data_setting, "val", val_preprocess)

    val_sampler = None
    is_shuffle = False
    batch_size = 4

    if engine.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        batch_size = 1
        is_shuffle = False

    val_loader = data.DataLoader(val_dataset,
                                 batch_size=batch_size,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 shuffle=is_shuffle,
                                 pin_memory=True,
                                 sampler=val_sampler)

    return val_loader, val_sampler
