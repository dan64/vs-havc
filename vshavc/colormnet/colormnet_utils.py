"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-09-14
version:
LastEditors: Dan64
LastEditTime: 2025-04-21
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Utility functions for the Vapoursynth wrapper of ColorMNet.
"""
import os
from os import path
import vapoursynth as vs
import numpy as np
from PIL import Image
import io
from vshavc.colormnet.dataset.range_transform import inv_im_trans, inv_lll2rgb_trans
from skimage import color
import cv2
import math
from vshavc.vsslib.constants import *
from vshavc.vsslib.vsutils import *


class RefImageReader:
    _instance = None
    use_all_refs: bool = True  # when true will be used all available reference frames
    ref_req_list_size: int = None
    num_ref_imgs: int = 0
    ref_last_idx: int = None
    ref_num_list: list[int] = None
    clip_total_frames: int = None
    clip_buffer_frames: int = None
    clip_last_frame: int = None
    clip_ref: vs.VideoNode = None
    clip_sc: vs.VideoNode = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, ref_list_size: int = DEF_NUM_XRF_FRAMES, use_all_refs: bool = True):
        self.use_all_refs = use_all_refs
        self.num_ref_imgs = 0
        self.ref_last_idx = 0
        # buffer size must be a multiple of 2
        self.ref_req_list_size = max(min(math.trunc(ref_list_size / 2) * 2, DEF_MAX_XRF_FRAMES), DEF_MIN_XRF_FRAMES)
        self.clip_total_frames: int = 0
        self.clip_buffer_size: int = 0
        self.clip_last_frame: int = 0

    def enabled(self):
        return self.use_all_refs

    def extend_clip_ref_list(self) -> bool:
        if self.clip_last_frame == self.clip_total_frames - 1:
            return False
        num_frames = min(self.clip_total_frames - self.clip_last_frame - 1, self.clip_buffer_size)
        batch_size = self.clip_last_frame + num_frames + 1
        num_ref_imgs = self.num_ref_imgs
        for i in range(self.clip_last_frame + 1, batch_size):
            frame = self.clip_sc.get_frame(i)
            if frame.props['_SceneChangePrev'] == 1:
                self.ref_num_list.append(i)
                self.num_ref_imgs += 1
        self.clip_last_frame = batch_size - 1
        self.num_ref_imgs = len(self.ref_num_list)
        return self.num_ref_imgs > num_ref_imgs

    def get_clip_ref_list(self, clip_sc: vs.VideoNode, start_frame: int = 0) -> int:
        # with self._vs_env.use():
        self.clip_sc = clip_sc
        self.ref_num_list = []
        self.clip_total_frames = clip_sc.num_frames
        start_frame = min(start_frame, self.clip_total_frames - 1)
        self.clip_buffer_size = min(self.clip_total_frames - start_frame, DEF_MAX_XREF_BUFFER)
        self.ref_req_list_size = min(self.clip_total_frames - start_frame, self.ref_req_list_size)

        for i in range(0, self.clip_buffer_size):
            frame = clip_sc.get_frame(i)
            if frame.props['_SceneChangePrev'] == 1:
                self.ref_num_list.append(start_frame + i)
                self.num_ref_imgs += 1
        self.clip_last_frame = start_frame + (self.clip_buffer_size - 1)
        for count in range(10):
            if self.num_ref_imgs < self.ref_req_list_size and self.clip_last_frame < (self.clip_total_frames - 1):
                self.extend_clip_ref_list()
            else:
                break
        if self.num_ref_imgs < DEF_MIN_RF_FRAMES:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            f"ColorMNet(): number of reference frames must be at least {DEF_MIN_RF_FRAMES}, found: ",
                            self.num_ref_imgs)
        return self.num_ref_imgs

    def reload_clip_ref(self, start_frame: int = 0):
        self.get_clip_ref_list(self.clip_ref, start_frame=start_frame)
        self.ref_last_idx = 0
        return self.num_ref_imgs

    def load_clip_ref(self, clip_ref: vs.VideoNode = None, clip_sc: vs.VideoNode = None, start_frame: int = 0):
        self.clip_ref = clip_ref
        if clip_sc is None:
            self.get_clip_ref_list(self.clip_ref, start_frame=start_frame)
        else:
            self.get_clip_ref_list(clip_sc, start_frame=start_frame)
        self.ref_last_idx = 0
        return self.num_ref_imgs

    """
    In the current implementation the reference frame is added when is loaded the clip frame with the same order.
    TODO: In ColorMNet it is possible to load in advanced also the future frames, they will be used
    when a correspondence will be found. It is possible to adopt the same strategy implemented in 
    DeepRemaster to load in advance a given number of reference frame.
    For example when is loaded the clip frame #1, could be possible to call the function set_ref_frame by passing
    the reference image of frame #100.  
    """

    def search_new_ref_imgs(self) -> bool:
        while not self.extend_clip_ref_list():
            if self.clip_last_frame == self.clip_total_frames - 1:
                return False
        return True

    def get_next_ref_frame(self, frame_n: int = 0) -> Image:

        if not self.use_all_refs:
            return None

        # extend the number of reference images
        if self.ref_last_idx >= (self.num_ref_imgs - 1) and self.clip_last_frame < self.clip_total_frames - 1:
            self.search_new_ref_imgs()

        if self.ref_last_idx > (self.num_ref_imgs - 1):
            return None  # no more reference frames are available

        # find the ref frame nearest to frame_n
        ref_half_idx = round(self.num_ref_imgs*0.5)
        if self.ref_last_idx > ref_half_idx:
            n_last = self.ref_last_idx
            while n_last > 0 and frame_n < self.ref_num_list[n_last]:
                n_last -= 1
            window = self.ref_last_idx - n_last

            if window < DEF_MAX_XREF_WINDOW:
                return None  # number of forward reference frames is enough

        n = self.ref_num_list[self.ref_last_idx]
        img = frame_to_image(self.clip_ref.get_frame(n))
        self.ref_last_idx += 1

        return img


def image_to_byte_array(img: Image, img_format: str = "jpeg", img_quality: int = 95) -> bytes:
    # BytesIO is a file-like buffer stored in memory
    img_byte_array = io.BytesIO()
    # image.save expects a file-like as an argument
    if img_format in ("jpg", "jpeg"):
        img.save(img_byte_array, format=img_format, subsampling=0, quality=img_quality)
    else:  # "png"
        img.save(img_byte_array, format=img_format)
    # Turn the BytesIO object back into a bytes object
    return img_byte_array.getvalue()


def byte_array_to_image(img_byte_array: bytes) -> Image:
    stream = io.BytesIO(img_byte_array)
    img = Image.open(stream).convert('RGB')
    return img


def detach_to_cpu(x):
    return x.detach().cpu()


def tensor_to_np_float(image):
    image_np = image.numpy().astype('float32')
    return image_np


def lab2rgb_transform_PIL(mask):
    mask_d = detach_to_cpu(mask)
    mask_d = inv_lll2rgb_trans(mask_d)
    im = tensor_to_np_float(mask_d)

    if len(im.shape) == 3:
        im = im.transpose((1, 2, 0))
    else:
        im = im[:, :, None]

    im = color.lab2rgb(im)

    return im.clip(0, 1)


def img_weighted_merge(img1: Image, img2: Image, weight: float = 0.5) -> Image:
    img1_np = np.asarray(img1)
    img2_np = np.asarray(img2)

    img_new = np.copy(img1_np)

    img_m = np.multiply(img1_np, 1 - weight) + np.multiply(img2_np, weight)
    img_m = np.uint8(np.clip(img_m, 0, 255))

    img_new[:, :, 0] = img_m[:, :, 0]
    img_new[:, :, 1] = img_m[:, :, 1]
    img_new[:, :, 2] = img_m[:, :, 2]

    return Image.fromarray(img_new)


def frm_to_img(frame: vs.VideoFrame) -> Image:
    np_array = np.dstack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return Image.fromarray(np_array, 'RGB')


def img_to_frm(img: Image, frame: vs.VideoFrame) -> vs.VideoFrame:
    np_array = np.array(img)
    [np.copyto(np.asarray(frame[plane]), np_array[:, :, plane]) for plane in range(frame.format.num_planes)]
    return frame
