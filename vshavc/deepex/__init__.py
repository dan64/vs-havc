"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-05-14
version:
LastEditors: Dan64
LastEditTime: 2026-05-09
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
main Vapoursynth wrapper for model: "Deep-Exemplar-based Video Colorization"
URL: https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization
"""
from __future__ import annotations, print_function

import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform_lib
from PIL import Image

import vapoursynth as vs
import vshavc.deepex.lib.TestTransforms as transforms
from vshavc.deepex.models.ColorVidNet import ColorVidNet
from vshavc.deepex.models.FrameColor import frame_colorization
from vshavc.deepex.models.NonlocalNet import VGG19_pytorch, WarpNet
from vshavc.deepex.utils.util import (batch_lab2rgb_transpose_mc, tensor_lab2rgb, uncenter_l)
from vshavc.deepex.utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

Tensor = torch.Tensor

import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Conversion from CIE-LAB,*?")

package_dir = os.path.dirname(os.path.realpath(__file__))


def deepex_colorizer(image_size: list = [432, 768], enable_resize: bool = False) -> ModelColorizer:
    return ModelColorizer(image_size=image_size, enable_resize=enable_resize, project_dir=package_dir)


def get_deepex_size(render_speed: str = 'medium', enable_resize: bool = False, ex_model: int = 1,
                    max_size: int = 0) -> list:
    if enable_resize:
        scale = 2
    else:
        scale = 1

    d_size = None

    if max_size == 0:
        max_size = 768

    if render_speed == 'auto':
        list_w = [256, 384, 512, 640]
        list_h = [144, 216, 288, 360]
        index_i = min(range(len(list_w)), key=lambda i: abs(list_w[i]*scale - max_size))
        d_size = [list_h[index_i]*scale, list_w[index_i]*scale]
        return d_size

    if ex_model in (0, 1, 3):
        match render_speed:
            case 'medium':
                d_size = [216 * scale, 384 * scale]
            case 'fast':
                d_size = [144 * scale, 256 * scale]
            case 'slow':
                d_size = [288 * scale, 512 * scale]
            case 'slower':
                d_size = [360 * scale, 640 * scale]
            case _:
                raise vs.Error("HAVC_deepex: unknown render_speed ->" + render_speed)
    else:
        match render_speed:
            case 'medium':
                d_size = [256 * scale, 256 * scale]
            case 'fast':
                d_size = [224 * scale, 224 * scale]
            case 'slow':
                d_size = [320 * scale, 320 * scale]
            case 'slower':
                d_size = [360 * scale, 360 * scale]
            case _:
                raise vs.Error("HAVC_deepex: unknown render_speed ->" + render_speed)

    return d_size


class ModelColorizer:
    _instance = None
    _initialized = False
    _frame_size = None
    IB_lab = None
    I_reference_lab = None
    features_B = None
    propagate = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, image_size: list = [216 * 2, 384 * 2], enable_resize: bool = False, project_dir: str = None):

        self.I_last_lab_predict = None
        self.enable_resize = enable_resize
        self.project_dir = project_dir

        if not self._initialized:
            self._frame_size = image_size
            self._colorize_model_init(image_size)
            self._initialized = True

    def set_ref_frame(self, frame_ref: Image = None, frame_propagate: bool = True):

        # new reference frame -> reset last prediction
        self.I_last_lab_predict = None
        self.propagate = frame_propagate

        IB_lab_large = self.transform(frame_ref).unsqueeze(0).cuda()
        if self.enable_resize:
            self.IB_lab = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear")
        else:
            self.IB_lab = IB_lab_large
        # IB_l = self.IB_lab[:, 0:1, :, :]
        # IB_ab = self.IB_lab[:, 1:3, :, :]
        with torch.no_grad():
            self.I_reference_lab = self.IB_lab
            I_reference_l = self.I_reference_lab[:, 0:1, :, :]
            I_reference_ab = self.I_reference_lab[:, 1:3, :, :]
            I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))
            self.features_B = self.vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)

    def colorize_frame(self, frame_i: Image = None, wls_filter_on: bool = True, render_vivid: bool = True) -> Image:

        # check if ref_frame was set, it should never happen but it could happen
        if (self.IB_lab is None) or (self.I_reference_lab is None) or (self.features_B is None):
            return frame_i

        if self.enable_resize:
            # save original frame
            np_frame_i = np.array(frame_i)
            frame_orig_size = frame_i.size

        # processing frame
        IA_lab_large = self.transform(frame_i).unsqueeze(0).cuda()
        if self.enable_resize:
            IA_lab = torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear")
        else:
            IA_lab = IA_lab_large
        # IA_l = IA_lab[:, 0:1, :, :]
        # IA_ab = IA_lab[:, 1:3, :, :]

        # frame_propagate can be set true if the new reference frame is the colored version
        # of the frame to be colored
        if self.I_last_lab_predict is None:
            if self.propagate:
                self.I_last_lab_predict = self.IB_lab  # last prediction -> reference frame
            else:
                self.I_last_lab_predict = torch.zeros_like(IA_lab).cuda()

        # start the frame colorization
        with torch.no_grad():
            I_current_lab = IA_lab
            I_current_ab_predict, I_current_nonlocal_lab_predict, features_current_gray = frame_colorization(
                I_current_lab,
                self.I_reference_lab,
                self.I_last_lab_predict,
                self.features_B,
                self.vggnet,
                self.nonlocal_net,
                self.colornet,
                feature_noise=0,
                temperature=1e-10,
            )
            # I_last_lab_predict = torch.cat((IA_l, I_current_ab_predict), dim=1)

        # upsampling
        curr_bs_l = IA_lab_large[:, 0:1, :, :]

        if render_vivid:
            # increase saturation by 25%
            if self.enable_resize:
                curr_predict = (torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2,
                                                                mode="bilinear") * 1.25)
            else:
                curr_predict = I_current_ab_predict.data.cpu() * 1.25
        else:
            if self.enable_resize:
                curr_predict = (
                    torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear"))
            else:
                curr_predict = I_current_ab_predict.data.cpu()

        # Weighted Least Square (wls) filtering algorithm provided by OpenCV
        # see: Fast Global Image Smoothing Based on Weighted Least Squares
        # http://publish.illinois.edu/visual-modeling-and-analytics/files/2014/10/FGS-TIP.pdf
        # parameters for wls filter
        lambda_value = 500
        sigma_color = 4

        if wls_filter_on:
            guide_image = uncenter_l(curr_bs_l) * 255 / 100
            wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
                guide_image[0, 0, :, :].cpu().numpy().astype(np.uint8), lambda_value, sigma_color
            )
            curr_predict_a = wls_filter.filter(curr_predict[0, 0, :, :].cpu().numpy())
            curr_predict_b = wls_filter.filter(curr_predict[0, 1, :, :].cpu().numpy())
            curr_predict_a = torch.from_numpy(curr_predict_a).unsqueeze(0).unsqueeze(0)
            curr_predict_b = torch.from_numpy(curr_predict_b).unsqueeze(0).unsqueeze(0)
            curr_predict_filter = torch.cat((curr_predict_a, curr_predict_b), dim=1)
            IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict_filter[:32, ...])
        else:
            IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict[:32, ...])

        if self.enable_resize:
            np_img_rgb = cv2.resize(IA_predict_rgb, frame_orig_size, interpolation=cv2.INTER_CUBIC)
            # restore original resolution
            np_img_rgb = self._chroma_np_post_process(np_img_rgb, np_frame_i)
        else:
            np_img_rgb = IA_predict_rgb

        # return the frame (clipping is performed on the conversion)
        return Image.fromarray(np_img_rgb, 'RGB').convert('RGB')

    def _chroma_np_post_process(self, img_np: np.ndarray, orig_np: np.ndarray) -> np.ndarray:
        img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
        # copy the chroma parametrs "U", "V", of "img_m" in "orig"
        orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)
        orig_copy = np.copy(orig_yuv)
        orig_copy[:, :, 1:3] = img_yuv[:, :, 1:3]
        return cv2.cvtColor(orig_copy, cv2.COLOR_YUV2RGB)

    def _colorize_model_init(self, image_size: list = [216 * 2, 384 * 2]):

        cudnn.benchmark = True

        self.transform = transforms.Compose(
            [CenterPad(image_size), transform_lib.CenterCrop(image_size),
             RGB2Lab(), ToTensor(), Normalize()])

        self.nonlocal_net = WarpNet(1)
        self.colornet = ColorVidNet(7)
        self.vggnet = VGG19_pytorch()
        vgg19_path = os.path.join(self.project_dir, "data/vgg19_conv.pth")
        self.vggnet.load_state_dict(torch.load(vgg19_path))

        for param in self.vggnet.parameters():
            param.requires_grad = False

        nonlocal_test_path = os.path.join(self.project_dir, "checkpoints/",
                                          "video_moredata_l1/nonlocal_net_iter_76000.pth")
        self.nonlocal_net.load_state_dict(torch.load(nonlocal_test_path))

        color_test_path = os.path.join(self.project_dir, "checkpoints/", "video_moredata_l1/colornet_iter_76000.pth")
        self.colornet.load_state_dict(torch.load(color_test_path))

        self.nonlocal_net.eval()
        self.colornet.eval()
        self.vggnet.eval()

        self.nonlocal_net.cuda()
        self.colornet.cuda()
        self.vggnet.cuda()
        self.I_last_lab_predict = None
