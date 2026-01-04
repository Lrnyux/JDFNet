
import logging
import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image
import math
import torch.nn.functional as F
import torch
import lpips
import torch.nn as nn
from matplotlib import pyplot as plt


def calc_rmse(a, b):
    return torch.sqrt(torch.mean(torch.pow(a - b, 2),dim=[1,2,3]))

def calc_psnr(a, b):
    mse = torch.mean((255 * a - 255 * b)**2,dim=[1,2,3])
    if torch.sum(mse) == 0:
        return float('inf')
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


def calc_ssim(a, b):

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(a)
    mu_y = nn.AvgPool2d(3, 1)(b)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(a * a) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(b * b) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(a * b) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.mean(SSIM,dim=[1,2,3])
    # return SSIM


class CustomFormatter(logging.Formatter):
    DATE = '\033[94m'
    GREEN = '\033[92m'
    WHITE = '\033[0m'
    WARNING = '\033[93m'
    RED = '\033[91m'

    def __init__(self):
        orig_fmt = "%(name)s: %(message)s"
        datefmt = "%H:%M:%S"
        super().__init__(orig_fmt, datefmt)

    def format(self, record):
        color = self.WHITE
        if record.levelno == logging.INFO:
            color = self.GREEN
        if record.levelno == logging.WARN:
            color = self.WARNING
        if record.levelno == logging.ERROR:
            color = self.RED
        self._style._fmt = "{}%(asctime)s {}[%(levelname)s]{} {}: %(message)s".format(
            self.DATE, color, self.DATE, self.WHITE)
        return logging.Formatter.format(self, record)


class ConsoleLogger():
    def __init__(self, training_type, phase='train'):
        super().__init__()
        self._logger = logging.getLogger(training_type)
        self._logger.setLevel(logging.INFO)
        formatter = CustomFormatter()
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.INFO)
        console_log.setFormatter(formatter)
        self._logger.addHandler(console_log)
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        self.logfile_dir = os.path.join('experiments/', training_type, time_str)
        os.makedirs(self.logfile_dir)
        logfile = os.path.join(self.logfile_dir, f'{phase}.log')
        file_log = logging.FileHandler(logfile, mode='a')
        file_log.setLevel(logging.INFO)
        file_log.setFormatter(formatter)
        self._logger.addHandler(file_log)

    def info(self, *args, **kwargs):
        """info"""
        self._logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        """warning"""
        self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        """error"""
        self._logger.error(*args, **kwargs)
        exit(-1)

    def getLogFolder(self):
        return self.logfile_dir


class AverageMeter():
    def __init__(self):
        self.val = 0
        self.count = 0
        self.sum = 0
        self.ave = 0


    def update(self, val, num=1):
        self.count = self.count + num
        self.val = val
        self.sum = self.sum + num * val
        self.ave = self.sum / self.count if self.count != 0 else 0.0


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]



