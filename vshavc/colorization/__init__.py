"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2025-01-03
version:
LastEditors: Dan64
LastEditTime: 2025-09-28
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
main Vapoursynth wrapper for model "Colorization", including:
"Colorful Image Colorization" (eccv16)
"Real-Time User-Guided Image Colorization with Learned Deep Priors" (siggraph17)
URL: https://github.com/richzhang/colorization
"""
from __future__ import annotations, print_function

import os

#import numpy as np
#import torch
#import torch.backends.cudnn as cudnn

from vsdeoldify.colorization.colorizers import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)


class ModelColorization:
    _instance = None
    _initialized = False
    _frame_size = None
    colorizer_eccv16 = None
    colorizer_siggraph17 = None
    colorizer_model = None

    def __new__(cls, *args, **kwargs):
         if cls._instance is None:
             cls._instance = super().__new__(cls)
         return cls._instance

    def __init__(self, model: str = 'siggraph17', use_gpu: bool = True):

        if not self._initialized:
            self.colorizer_model = model
            self.use_gpu = use_gpu
            self._colorize_init()
            self._initialized = True
        else:
            if self.colorizer_model != model:
                self.colorizer_model = model
                self._colorize_init()

    def colorize_frame_ext(self, frame_i: np.ndarray = None, f_size: int = 256) -> np.ndarray:

        img = load_img_rgb(frame_i)
        # grab L channel in both original ("orig") and resized ("rs") resolutions
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(f_size, f_size))
        if self.use_gpu:
            tens_l_rs = tens_l_rs.cuda()

        # colorizer outputs ab map
        # resize and concatenate to original L channel

        if self.colorizer_model == 'siggraph17':
            np_img_float = postprocess_tens(tens_l_orig, self.colorizer_siggraph17(tens_l_rs).cpu())
        else:
            np_img_float = postprocess_tens(tens_l_orig, self.colorizer_eccv16(tens_l_rs).cpu())

        # return the frame converted in np.ndarray(uint8)
        np_img_rgb = np.uint8(np.clip(np_img_float * 255, 0, 255))
        return np_img_rgb

    def colorize_frame(self, frame_i: np.ndarray = None) -> np.ndarray:

        img = load_img_rgb(frame_i)
        # default size to process images is 256x256
        # grab L channel in both original ("orig") and resized ("rs") resolutions
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        if self.use_gpu:
            tens_l_rs = tens_l_rs.cuda()

        # colorizer outputs 256x256 ab map
        # resize and concatenate to original L channel

        if self.colorizer_model == 'siggraph17':
            np_img_float = postprocess_tens(tens_l_orig, self.colorizer_siggraph17(tens_l_rs).cpu())
        else:
            np_img_float = postprocess_tens(tens_l_orig, self.colorizer_eccv16(tens_l_rs).cpu())

        # return the frame converted in np.ndarray(uint8)
        np_img_rgb = np.uint8(np.clip(np_img_float * 255, 0, 255))
        return np_img_rgb

    def _colorize_init(self):

        if self.colorizer_model == 'siggraph17':
            self.colorizer_siggraph17 = siggraph17(pretrained=True).eval()
            if self.use_gpu:
                self.colorizer_siggraph17.cuda()
        else:
            self.colorizer_eccv16 = eccv16(pretrained=True).eval()
            if self.use_gpu:
                self.colorizer_eccv16.cuda()




