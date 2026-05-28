"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-06-21
version:
LastEditors: Dan64
LastEditTime: 2025-01-25
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
functions utility for Deep Remaster.
"""

import os
from torchvision import transforms
from skimage import color
import numpy as np
import cv2
from PIL import Image
import vapoursynth as vs
import logging

core = vs.core

_IMG_EXTENSIONS = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG',
                   '.ppm', '.PPM', '.bmp', '.BMP']


def convertLAB2RGB(lab):
    lab[:, :, 0:1] = lab[:, :, 0:1] * 100  # [0, 1] -> [0, 100]
    lab[:, :, 1:3] = np.clip(lab[:, :, 1:3] * 255 - 128, -100, 100)  # [0, 1] -> [-128, 128]
    rgb = color.lab2rgb(lab.astype(np.float64))
    return rgb


def convertRGB2LABTensor(rgb):
    lab = color.rgb2lab(np.asarray(rgb))  # RGB -> LAB L[0, 100] a[-127, 128] b[-128, 127]
    ab = np.clip(lab[:, :, 1:3] + 128, 0, 255)  # AB --> [0, 255]
    ab = transforms.ToTensor()(ab) / 255.
    L = lab[:, :, 0] * 2.55  # L --> [0, 255]
    L = Image.fromarray(np.uint8(L))
    L = transforms.ToTensor()(L)  # tensor [C, H, W]
    return L, ab.float()


def addMergin(img, target_w, target_h, background_color=(0, 0, 0)):
    width, height = img.size
    if width == target_w and height == target_h:
        return img
    scale = max(target_w, target_h) / max(width, height)
    width = int(width * scale / 16.) * 16
    height = int(height * scale / 16.) * 16
    img = transforms.Resize((height, width), interpolation=Image.BICUBIC)(img)

    xp = (target_w - width) // 2
    yp = (target_h - height) // 2
    result = Image.new(img.mode, (target_w, target_h), background_color)
    result.paste(img, (xp, yp))
    return result


def frame_to_image(frame: vs.VideoFrame) -> Image:
    npArray = np.dstack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return Image.fromarray(npArray, 'RGB')


def image_to_frame(img: Image, frame: vs.VideoFrame) -> vs.VideoFrame:
    npArray = np.array(img)
    [np.copyto(np.asarray(frame[plane]), npArray[:, :, plane]) for plane in range(frame.format.num_planes)]
    return frame


def frame_to_np_array(frame: vs.VideoFrame) -> np.ndarray:
    npArray = np.dstack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return npArray


def np_array_to_frame(npArray: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
    [np.copyto(np.asarray(frame[plane]), npArray[:, :, plane]) for plane in range(frame.format.num_planes)]
    return frame


def disable_warnings():
    logger_blocklist = [
        "matplotlib",
        "PIL",
    ]

    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)


def is_img_file(dir="./", fname: str = "") -> bool:
    filename = os.path.join(dir, fname)

    if not os.path.isfile(filename):
        return False

    return any(fname.endswith(extension) for extension in _IMG_EXTENSIONS)


def is_ref_file(dir="./", fname: str = "") -> bool:
    filename = os.path.join(dir, fname)

    if not os.path.isfile(filename):
        return False

    return fname.startswith("ref_") and any(fname.endswith(extension) for extension in _IMG_EXTENSIONS)


def get_ref_num(filename: str = ""):
    fname = filename.split(".")[0]
    fnum = int(fname.split("_")[-1])
    return fnum


def get_ref_names(dir="./") -> list:
    img_ref_list = [f for f in os.listdir(dir) if is_ref_file(dir, f)]
    return img_ref_list


def get_ref_images(img_dir="./") -> list:
    img_ref_file = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if is_img_file(img_dir, f)]
    return img_ref_file


def get_ref_list(img_dir="./") -> tuple[list, list]:
    img_ref_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if is_img_file(img_dir, f)]
    img_ref_list.sort()
    ref_num_list = [get_ref_num(f) for f in img_ref_list]
    return img_ref_list, ref_num_list


def resize_for_inference(clip: vs.VideoNode = None, frame_mindim: int = 320) -> vs.VideoNode:
    v_w = clip.width
    v_h = clip.height
    minwh = min(v_w, v_h)
    scale = 1
    if minwh != frame_mindim:
        scale = frame_mindim / minwh
    frame_w = round(v_w * scale / 16.) * 16
    frame_h = round(v_h * scale / 16.) * 16
    return clip.resize.Spline64(width=frame_w, height=frame_h)


def vs_recover_resolution(orig: vs.VideoNode = None, clip: vs.VideoNode = None) -> vs.VideoNode:
    def copy_luma_frame(n, f):
        img_orig = frame_to_image(f[0])
        img_clip = frame_to_image(f[1])
        img_m = chroma_restore(img_clip, img_orig)
        return image_to_frame(img_m, f[0].copy())

    clip = clip.std.ModifyFrame(clips=[orig, clip], selector=copy_luma_frame)

    return clip


def chroma_restore(img_m: Image, orig: Image) -> Image:
    img_np = np.asarray(img_m)
    orig_np = np.asarray(orig)
    img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)
    orig_copy = np.copy(orig_yuv)
    orig_copy[:, :, 1:3] = img_yuv[:, :, 1:3]
    img_np_new = cv2.cvtColor(orig_copy, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img_np_new)
