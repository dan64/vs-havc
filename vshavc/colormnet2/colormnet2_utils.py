"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2025-09-28
version:
LastEditors: Dan64
LastEditTime: 2026-04-25
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Utility functions for the Vapoursynth wrapper of CMNET2.
"""
import os
from os import path
import vapoursynth as vs
import numpy as np
from PIL import Image
import io
from vshavc.colormnet2.dataset.range_transform import inv_im_trans, inv_lll2rgb_trans
from skimage import color
import cv2
import math
from vshavc.vsslib.constants import *
from vshavc.vsslib.vsutils import *

_IMG_EXTENSIONS = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG',
                   '.ppm', '.PPM', '.bmp', '.BMP']

class RefImageReader2:
    _instance = None
    ref_req_list_size: int = None
    num_ref_imgs: int = 0
    ref_num_list: list[int] = None
    clip_total_frames: int = None
    clip_buffer_frames: int = None
    clip_last_frame: int = None
    clip_ref: vs.VideoNode = None
    clip_sc: vs.VideoNode = None
    source_mode: str = "vs"  # "vs" (default) or "dir"
    ref_img_list: list[str] = None  # only used when source_mode == "dir"
    target_size: tuple[int, int] = None  # (W, H) to resize disk-loaded refs; None = no resize

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, ref_list_size: int = DEF_XRF_WINDOW_SIZE):
        self.num_ref_imgs = 0
        # buffer size must be a multiple of 2
        self.ref_req_list_size = max(min(math.trunc(ref_list_size / 2) * 2, DEF_MAX_XRF_FRAMES), DEF_MIN_XRF_FRAMES)
        self.clip_total_frames: int = 0
        self.clip_buffer_size: int = 0
        self.clip_last_frame: int = 0

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

    def get_clip_ref_list(self, clip_sc: vs.VideoNode, start_frame: int = 0, window_size: int = None) -> int:
        self.clip_sc = clip_sc
        self.ref_num_list = []
        self.clip_total_frames = clip_sc.num_frames
        start_frame = min(start_frame, self.clip_total_frames - 1)
        self.clip_buffer_size = min(self.clip_total_frames - start_frame, DEF_MAX_XREF_BUFFER)
        req_size = window_size if window_size is not None else self.ref_req_list_size
        self.ref_req_list_size = min(self.clip_total_frames - start_frame, req_size)

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
        if self.num_ref_imgs < 1:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            f"CMNET2: number of reference frames must be at least 1")
        return self.num_ref_imgs

    def reload_clip_ref(self, start_frame: int = 0):
        self.get_clip_ref_list(self.clip_sc, start_frame=start_frame)
        return self.num_ref_imgs

    def load_clip_ref(self, clip_ref: vs.VideoNode = None, clip_sc: vs.VideoNode = None,
                      start_frame: int = 0, window_size: int = None):
        self.clip_ref = clip_ref
        sc = clip_sc if clip_sc is not None else clip_ref
        self.get_clip_ref_list(sc, start_frame=start_frame, window_size=window_size)
        return self.num_ref_imgs

    def load_from_dir(self, sc_framedir: str, target_size: tuple[int, int] = None) -> int:
        """
        Loads reference frames directly from a filesystem directory.
        Filenames must follow the pattern ref_nnnnnn.[jpg|png] where nnnnnn
        is the frame number in the video. Bypasses VS pipeline re-evaluation.

        target_size: (W, H) to resize each ref when read. If None, refs are
        loaded at their native size (caller is responsible for consistency
        with the B&W clip being colorized).
        """
        self.source_mode = "dir"
        self.target_size = target_size
        self.ref_img_list, self.ref_num_list = get_ref_list(sc_framedir)
        self.num_ref_imgs = len(self.ref_img_list)
        if self.num_ref_imgs < 1:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "RefImageReader2.load_from_dir(): at least 1 reference frames required, found: ",
                            self.num_ref_imgs)
        return self.num_ref_imgs

    def get_ref_image(self, idx: int) -> Image:
        """Pure accessor — returns the reference image at position idx without modifying any state."""
        if self.source_mode == "dir":
            img = Image.open(self.ref_img_list[idx]).convert('RGB')
            if self.target_size is not None and img.size != self.target_size:
                img = img.resize(self.target_size, Image.Resampling.LANCZOS)
            return img
        n = self.ref_num_list[idx]
        return frame_to_image(self.clip_ref.get_frame(n))

    def extend_if_needed(self, required_idx: int) -> bool:
        """Extends the ref list until num_ref_imgs > required_idx or the clip is exhausted."""
        if self.source_mode == "dir":
            return required_idx < self.num_ref_imgs
        while self.num_ref_imgs <= required_idx:
            if not self.extend_clip_ref_list():
                return False
        return True


class PermMemWindow:
    """Orchestrates the sliding permanent-memory window for CMNET2."""

    def __init__(self, colorizer, reader: RefImageReader2, window_size: int):
        self.colorizer = colorizer
        self.reader = reader
        self.window_size = min(window_size, reader.num_ref_imgs)
        self.slide_step = max(1, round(self.window_size * DEF_XRF_SLIDE_PERCENT + 0.5))  # reserved for future use
        self.ref_half_idx = max(0, round(self.window_size * (1 - DEF_FUTURE_FRAME_WEIGHT)) - 1)
        self.next_ref_idx = 0
        self.activation_frame = None

    def preload_initial(self):
        """Load the first window_size reference images into permanent memory before the colorization loop."""
        for i in range(self.window_size):
            self.colorizer.preload_reference(self.reader.get_ref_image(i))
        self.next_ref_idx = self.window_size
        self.activation_frame = self.reader.ref_num_list[self.ref_half_idx]

    def adjust(self, frame_n: int):
        """Slide the permanent-memory window forward by one step when frame_n passes activation_frame."""
        if self.activation_frame is None:
            return
        if frame_n <= self.activation_frame:
            return
        if self.next_ref_idx >= self.reader.num_ref_imgs:
            if not self.reader.extend_if_needed(self.next_ref_idx):
                return
        self.colorizer.slide_permanent_memory(1)
        self.colorizer.preload_reference(self.reader.get_ref_image(self.next_ref_idx))
        self.next_ref_idx += 1
        self.ref_half_idx += 1
        self.activation_frame = self.reader.ref_num_list[self.ref_half_idx]

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

def get_ref_list(img_dir="./") -> tuple[list, list]:
    img_ref_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if is_img_file(img_dir, f)]
    img_ref_list.sort()
    ref_num_list = [get_ref_num(f) for f in img_ref_list]
    return img_ref_list, ref_num_list


def is_img_file(dir="./", fname: str = "") -> bool:
    filename = os.path.join(dir, fname)

    if not os.path.isfile(filename):
        return False

    return any(fname.endswith(extension) for extension in _IMG_EXTENSIONS)

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
