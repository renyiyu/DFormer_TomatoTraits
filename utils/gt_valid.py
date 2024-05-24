import math
import os
import sys
import glob
import json
import cv2
import torch
import torch.nn as nn
from torch.utils import data
import random
import argparse
import numpy as np

import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from utils.dataloader.dataloader import ValPre, get_val_loader
from models.builder import EncoderDecoder as segmodel
from utils.engine.engine import Engine
from utils.engine.logger import get_logger

from utils.dataloader.RGBXDataset import RGBXDataset


def load_data(name, rgb_dir, x_dir, config):
    # load rgb  nad depth data
    rgb_path = os.path.join(rgb_dir, name + '.jpg')
    rgb = np.array(cv2.imread(rgb_path, cv2.COLOR_BGR2RGB))

    x_path = os.path.join(x_dir, name + '.png')
    x = np.array(cv2.imread(x_path, cv2.IMREAD_GRAYSCALE))
    x = cv2.merge([x, x, x])

    # preprocess rgb and depth data
    preprocess = ValPre(config.rgb_norm_mean,
                        config.rgb_norm_std,
                        config.x_norm_mean,
                        config.x_norm_std,
                        config.x_is_single_channel,
                        config)
    rgb, x = preprocess(rgb, x)

    # to tensor
    rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
    x = torch.from_numpy(np.ascontiguousarray(x)).float()

    return rgb, x


def load_model(config, weight_path):
    criterion = nn.MSELoss(reduction="mean", )
    model = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d)
    weight = torch.load(weight_path)["model"]

    print("load model")
    model.load_state_dict(weight)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model


def norm_param(gt_label_filepath):
    with open(gt_label_filepath) as json_file:
        gt_label_file = json.load(json_file)
    gt_values = torch.tensor(
        [[v for k, v in traits.items() if not np.isnan(v)] for name, traits in gt_label_file.items()])
    gt_mean = torch.mean(gt_values, dim=0)
    gt_std = torch.std(gt_values, dim=0)

    return gt_mean, gt_std


def infer(name, model, gt_train_label_filepath, rgb_dir, x_dir, config):
    gt_mean, gt_std = norm_param(gt_train_label_filepath)
    gt_mean = gt_mean.cuda(non_blocking=True)
    gt_std = gt_std.cuda(non_blocking=True)
    with torch.no_grad():
        model.eval()
        device = torch.device("cuda")
        rgb, x = load_data(name, rgb_dir, x_dir, config)

        rgb = rgb.cuda(non_blocking=True)
        x = x.cuda(non_blocking=True)

        out = model(rgb, x)
        out_denorm = out * gt_std + gt_mean

    return out_denorm[0].cpu().numpy().tolist()


def calculate_error(pred_dict, gt_dict):
    n = len(gt_dict)
    traits = ['height', 'fw_plant', 'leaf_area', 'number_of_red_fruits']
    error = 0
    for trait in traits:
        diff = []
        for name, value_dict in pred_dict.items():
            if name in gt_dict.keys():
                pred = pred_dict[name][trait]
                gt = gt_dict[name][trait]
                if name == "number_of_red_fruits" and pred < 0:
                    pred = 0
                re = (pred - gt) / (gt + 1)
                diff.append(math.pow(re, 2))
        error += np.sqrt(np.nanmean(diff))
    error /= len(traits)
    return error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    logger = get_logger()
    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        # args.config = 'local_configs.TomatoPartA.DFormer_Tiny'
        args.config = 'local_configs.TomatoPartA.DFormer_Small'
        exec("from " + args.config + " import config")
        if "x_modal" not in config:
            config["x_modal"] = "d"

        weight_path = 'checkpoints/dformer_small_final.pth'
        model = load_model(config, weight_path=weight_path)

        rgb_dir = 'datasets/val3_test/RGB'
        x_dir = 'datasets/val3_test/Depth'
        pred_path = 'datasets/val3_test/pred1.json'
        train_label_filepath = 'datasets/val3_test/label.json'

        # read existing json
        with open(pred_path, 'w') as f:
            pred_dict = {}
            img_paths = glob.glob(os.path.join(rgb_dir, '*.jpg'))
            for img_path in img_paths:
                basename = os.path.basename(img_path).split('.')[0]
                print(basename)
                pred_dict[basename] = {}
                pred = infer(basename, model, train_label_filepath, rgb_dir, x_dir, config)
                pred_dict[basename]['height'] = pred[0]
                pred_dict[basename]['fw_plant'] = pred[1]
                pred_dict[basename]['leaf_area'] = pred[2]
                if pred[3] < 0:
                    pred_dict[basename]['number_of_red_fruits'] = 0
                else:
                    pred_dict[basename]['number_of_red_fruits'] = pred[3]
            json.dump(pred_dict, f)

        # gt_label_filepath = 'datasets/val3_test/gt.json'
        # with open(gt_label_filepath) as json_file:
        #     gt_dict = json.load(json_file)
        # error = calculate_error(pred_dict, gt_dict)
        # print("RMSRE: ", error)










