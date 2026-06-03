"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-06-21
version:
LastEditors: Dan64
LastEditTime: 2026-05-19
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
DeepRemaster rendering class.
"""

import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import math
import os
import sys
import vapoursynth as vs
from vshavc.remaster.remaster_utils import *
from vshavc.vsslib.constants import *
from vshavc.vsslib.vsutils import *

Tensor = torch.Tensor

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# configuring torch
torch.backends.cudnn.benchmark = True

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model")

"""
------------------------------------------------------------------------------- 
Class: RemasterColorizer
------------------------------------------------------------------------------- 
Description: 
Class to perform colorization with DeepRemaster using Vapoursynth Video Frame 
to access the reference frames.
------------------------------------------------------------------------------- 
"""


class RemasterColorizer:
    _instance = None
    _initialized = False
    _frame_size = None
    # _vs_env = None
    clip_ref: vs.VideoNode = None
    clip_sc: vs.VideoNode = None
    device = None
    modelC = None
    refstorage: Tensor = None
    refimgs: Tensor = None
    target_w: int = None
    target_h: int = None
    num_ref_imgs: int = 0
    ref_minedge: int = None
    ref_buffer_size: int = None
    ref_storage_size: int = None
    ref_last_idx: int = None
    ref_half_idx: int = None
    refs: list[Image] = None
    ref_num_list: list[int] = None
    fast_refs: bool = False
    ref_step: int = 0
    max_buffer_size: int = DEF_MAX_BUFFER_SIZE
    clip_total_frames: int = None
    clip_buffer_frames: int = None
    clip_last_frame: int = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, clip_ref: vs.VideoNode = None, ref_minedge: int = 256,
                 ref_buffer_size: int = 20, device_index: int = 0, ref_step: int = 0,
                 model_dir: str = model_dir):

        # Environment-object representing the environment the script is currently running in
        # self._vs_env = vs.get_current_environment()
        self.clip_ref = clip_ref
        if device_index == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda", device_index)
        self.num_ref_imgs = 0
        self.ref_storage_size = 0
        self.ref_last_idx = 0
        self.ref_half_idx = 0
        self.ref_minedge = ref_minedge
        self.fast_refs = 1 < ref_step < 5
        self.ref_step = ref_step
        # buffer size must be a multiple of 2
        self.ref_buffer_size = max(min(math.trunc(ref_buffer_size / 2) * 2, DEF_MAX_RF_FRAMES), DEF_MIN_RF_FRAMES)
        self.max_buffer_size = DEF_MAX_BUFFER_SIZE
        self.clip_total_frames: int = 0
        self.clip_buffer_size: int = 0
        self.clip_last_frame: int = 0
        if not self._initialized:
            self.model_load(model_dir)
            self._initialized = True

    def model_load(self, model_dir: str = None):

        model_path = os.path.join(model_dir, 'remasternet.pth.tar')
        state_dict = torch.load(model_path)

        self.modelC = __import__('vshavc.remaster.model.remasternet', fromlist=['NetworkC']).NetworkC()
        self.modelC.load_state_dict(state_dict['modelC'])
        self.modelC = self.modelC.to(self.device)
        self.modelC.eval()

    def extend_clip_ref_list(self):
        if self.clip_last_frame == self.clip_total_frames - 1:
            return
        num_frames = min(self.clip_total_frames - self.clip_last_frame - 1, self.max_buffer_size)
        batch_size = self.clip_last_frame + num_frames + 1
        for i in range(self.clip_last_frame + 1, batch_size):
            if self.fast_refs:
                if i % self.ref_step == 0:
                    self.ref_num_list.append(i)
                    self.num_ref_imgs += 1
            else:
                frame = self.clip_sc.get_frame(i)
                if frame.props['_SceneChangePrev'] == 1:
                    self.ref_num_list.append(i)
                    self.num_ref_imgs += 1
        self.clip_last_frame = batch_size - 1

    def get_clip_ref_list(self, clip_sc: vs.VideoNode) -> int:
        # with self._vs_env.use():
        self.clip_sc = clip_sc
        self.ref_num_list = []
        self.clip_total_frames = clip_sc.num_frames
        self.clip_buffer_size = min(self.clip_total_frames, self.max_buffer_size)
        for i in range(0, self.clip_buffer_size):
            if self.fast_refs:
                if i % self.ref_step == 0:
                    self.ref_num_list.append(i)
                    self.num_ref_imgs += 1
            else:
                frame = clip_sc.get_frame(i)
                if frame.props['_SceneChangePrev'] == 1:
                    self.ref_num_list.append(i)
                    self.num_ref_imgs += 1
        self.clip_last_frame = self.clip_buffer_size - 1
        for count in range(10):
            if self.num_ref_imgs < self.ref_buffer_size and self.clip_last_frame < (self.clip_total_frames - 1):
                self.extend_clip_ref_list()
            else:
                break
        if self.num_ref_imgs < DEF_MIN_RF_FRAMES:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            f"RemasterColorizer(): number of reference frames must be at least {DEF_MIN_RF_FRAMES}, found: ",
                            self.num_ref_imgs)
        return self.num_ref_imgs

    def load_clip_ref(self, clip_sc: vs.VideoNode = None):
        if clip_sc is None:
            self.get_clip_ref_list(self.clip_ref)
        else:
            self.get_clip_ref_list(clip_sc)
        self.ref_storage_size = min(self.ref_buffer_size, self.num_ref_imgs)

        if self.ref_storage_size == 0:
            return 0

        # with self._vs_env.use():
        self.ref_half_idx = round(self.ref_storage_size * (1 - DEF_FUTURE_FRAME_WEIGHT)) - 1
        n = self.ref_num_list[0]
        img = frame_to_image(self.clip_ref.get_frame(n))
        w, h = img.size
        aspect_mean = w / h
        self.target_w = int(self.ref_minedge * aspect_mean) if aspect_mean > 1 else self.ref_minedge
        self.target_h = self.ref_minedge if aspect_mean >= 1 else int(self.ref_minedge / aspect_mean)
        # initialize the reference images Tensor storage
        self.refstorage = torch.FloatTensor(self.ref_storage_size, 3, self.target_h, self.target_w)
        img = addMergin(img, target_w=self.target_w, target_h=self.target_h)
        # add first reference frames
        self.refstorage[0] = transforms.ToTensor()(img)
        # add the remaining reference frames
        for i in range(1, self.ref_storage_size):
            n = self.ref_num_list[i]
            img = frame_to_image(self.clip_ref.get_frame(n))
            img = addMergin(img, target_w=self.target_w, target_h=self.target_h)
            self.refstorage[i] = transforms.ToTensor()(img)
        self.ref_last_idx = self.ref_storage_size - 1
        # create a view on the torch storage
        self.refimgs = self.refstorage.view(1, self.refstorage.size(0), self.refstorage.size(1),
                                            self.refstorage.size(2),
                                            self.refstorage.size(3)).to(self.device)
        return self.num_ref_imgs

    def ref_buffer_adjust(self, frame_n: int = 0):

        if frame_n >= (self.clip_last_frame - self.clip_buffer_size):
            self.extend_clip_ref_list()

        if self.ref_last_idx == (self.num_ref_imgs - 1):
            return  # nothing to do

        if frame_n <= self.ref_num_list[self.ref_half_idx]:
            return  # do nothing

        # shift by 1 position on the left the reference images stored in the tensor array
        for i in range(0, self.ref_storage_size - 1):
            self.refstorage[i] = self.refstorage[i + 1]

        # now add a new reference image
        # with self._vs_env.use():
        self.ref_last_idx += 1
        self.ref_half_idx += 1
        n = self.ref_num_list[self.ref_last_idx]

        img = frame_to_image(self.clip_ref.get_frame(n))
        img = addMergin(img, target_w=self.target_w, target_h=self.target_h)
        # add the new reference image to the storage
        self.refstorage[self.ref_storage_size - 1] = transforms.ToTensor()(img)

        # update tensor view
        self.refimgs = self.refstorage.view(1, self.refstorage.size(0), self.refstorage.size(1),
                                            self.refstorage.size(2), self.refstorage.size(3)).to(self.device)

    def resize(self, img: Image, width: int = None, height: int = None) -> Image:
        return addMergin(img, target_w=width, target_h=height)

    def process_frames(self, frames: list[Image] = None, last_frame_idx: int = 0,
                       convert_to_pil: bool = False) -> list[Image]:
        # Process
        with torch.no_grad():
            t_input = None
            self.ref_buffer_adjust(last_frame_idx)
            nframes = len(frames)
            for i in range(nframes):
                i_frame = frames[i]
                frame_l = cv2.cvtColor(i_frame, cv2.COLOR_RGB2GRAY)
                frame_l = torch.from_numpy(frame_l).view(frame_l.shape[0], frame_l.shape[1], 1)
                frame_l = frame_l.permute(2, 0, 1).float()  # HWC to CHW
                frame_l /= 255.
                frame_l = frame_l.view(1, frame_l.size(0), 1, frame_l.size(1), frame_l.size(2))
                t_input = frame_l if i == 0 else torch.cat((t_input, frame_l), 2)

            if nframes == 1:
                # add the same frame, because the number of frames for inference must be at least = 2
                t_input = torch.cat((t_input, frame_l), 2)

            output_l = t_input.to(self.device)

            out_frames: list[Image] = []

            if self.refimgs is None:
                output_ab = self.modelC(output_l)
            else:
                output_ab = self.modelC(output_l, self.refimgs)
            output_l = output_l.detach().cpu()
            output_ab = output_ab.detach().cpu()
            for i in range(nframes):
                out_l = output_l[0, :, i, :, :]
                out_c = output_ab[0, :, i, :, :]
                output = torch.cat((out_l, out_c), dim=0).numpy().transpose((1, 2, 0))
                if convert_to_pil:
                    output = Image.fromarray(np.uint8(convertLAB2RGB(output) * 255))
                else:
                    output = np.uint8(convertLAB2RGB(output) * 255)
                out_frames.append(output)

            torch.cuda.empty_cache()
            return out_frames


"""
------------------------------------------------------------------------------- 
Class: RemasterEngine
------------------------------------------------------------------------------- 
Description: 
Class to perform colorization with DeepRemaster using direct access to the 
reference frames folder.
------------------------------------------------------------------------------- 
"""


class RemasterEngine:
    _instance = None
    _initialized = False
    device = None
    modelC = None
    refstorage: Tensor = None
    refimgs: Tensor = None
    target_w: int = None
    target_h: int = None
    ref_storage_size: int = None
    num_ref_imgs: int = 0
    ref_last_idx: int = None
    ref_half_idx: int = None
    ref_minedge: int = None
    ref_buffer_size: int = None
    refs: list[Image] = None
    ref_img_list: list[str] = None
    ref_num_list: list[int] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, device_index: int = 0,
                 ref_minedge: int = 256,
                 ref_buffer_size: int = 20,
                 model_dir: str = model_dir):

        if device_index == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda", device_index)
        self.ref_minedge = ref_minedge
        # buffer size must be a multiple of 2
        self.ref_buffer_size = max(min(math.trunc(ref_buffer_size / 2) * 2, DEF_MAX_RF_FRAMES), DEF_MIN_RF_FRAMES)
        self.num_ref_imgs = 0
        self.ref_storage_size = 0
        self.ref_last_idx = 0
        self.ref_half_idx = 0

        if not self._initialized:
            self.model_load(model_dir)
            self._initialized = True

    def model_load(self, model_dir: str = None):

        model_path = os.path.join(model_dir, 'remasternet.pth.tar')
        state_dict = torch.load(model_path)

        self.modelC = __import__('vshavc.remaster.model.remasternet', fromlist=['NetworkC']).NetworkC()
        self.modelC.load_state_dict(state_dict['modelC'])
        self.modelC = self.modelC.to(self.device)
        self.modelC.eval()

    def load_ref_dir(self, rf_dir: str = None) -> int:
        self.ref_img_list, self.ref_num_list = get_ref_list(rf_dir)
        self.num_ref_imgs = len(self.ref_img_list)
        # number of reference images must be at least 2
        if self.num_ref_imgs < 2:
            self.ref_num_list.append(self.ref_num_list[self.num_ref_imgs - 1])
            self.ref_img_list.append(self.ref_img_list[self.num_ref_imgs - 1])
            self.num_ref_imgs += 1
        self.load_ref_buffer()
        return self.num_ref_imgs

    def load_ref_buffer(self) -> int:
        self.ref_storage_size = min(self.ref_buffer_size, self.num_ref_imgs)

        if self.ref_storage_size == 0:
            return 0

        self.ref_half_idx = round(self.ref_storage_size * (1 - DEF_FUTURE_FRAME_WEIGHT)) - 1
        f_img = self.ref_img_list[0]
        img = Image.open(f_img).convert('RGB')
        w, h = img.size
        aspect_mean = w / h
        self.target_w = int(self.ref_minedge * aspect_mean) if aspect_mean > 1 else self.ref_minedge
        self.target_h = self.ref_minedge if aspect_mean >= 1 else int(self.ref_minedge / aspect_mean)
        # initialize the reference images Tensor storage
        self.refstorage = torch.FloatTensor(self.ref_storage_size, 3, self.target_h, self.target_w)
        # first reference frame
        img = addMergin(img, target_w=self.target_w, target_h=self.target_h)
        self.refstorage[0] = transforms.ToTensor()(img)
        # add the remaining reference frames
        for i in range(1, self.ref_storage_size):
            f_img = self.ref_img_list[i]
            img = Image.open(f_img).convert('RGB')
            img = addMergin(img, target_w=self.target_w, target_h=self.target_h)
            self.refstorage[i] = transforms.ToTensor()(img)
        self.ref_last_idx = self.ref_storage_size - 1
        # create a view on the torch storage
        self.refimgs = self.refstorage.view(1, self.refstorage.size(0), self.refstorage.size(1),
                                            self.refstorage.size(2), self.refstorage.size(3)).to(self.device)
        return self.num_ref_imgs

    def ref_buffer_adjust(self, frame_n: int = 0):
        if self.ref_last_idx == (self.num_ref_imgs - 1):
            return  # nothing to do

        if frame_n <= self.ref_num_list[self.ref_half_idx]:
            return  # do nothing

            # shift by 1 position on the left the reference images stored in the tensor array
        for i in range(0, self.ref_storage_size - 1):
            self.refstorage[i] = self.refstorage[i + 1]

        self.ref_last_idx += 1
        self.ref_half_idx += 1
        f_img = self.ref_img_list[self.ref_last_idx]
        img = Image.open(f_img).convert('RGB')
        img = addMergin(img, target_w=self.target_w, target_h=self.target_h)
        # add the new reference image to the storage
        self.refstorage[self.ref_storage_size - 1] = transforms.ToTensor()(img)

        # update tensor view
        self.refimgs = self.refstorage.view(1, self.refstorage.size(0), self.refstorage.size(1),
                                            self.refstorage.size(2), self.refstorage.size(3)).to(self.device)

    def process_frames(self, frames: list[Image] = None, last_frame_idx: int = 0,
                       convert_to_pil: bool = False) -> list[Image]:
        # Process
        with torch.no_grad():
            t_input = None
            self.ref_buffer_adjust(last_frame_idx)
            nframes = len(frames)
            for i in range(nframes):
                i_frame = frames[i]
                frame_l = cv2.cvtColor(i_frame, cv2.COLOR_RGB2GRAY)
                frame_l = torch.from_numpy(frame_l).view(frame_l.shape[0], frame_l.shape[1], 1)
                frame_l = frame_l.permute(2, 0, 1).float()  # HWC to CHW
                frame_l /= 255.
                frame_l = frame_l.view(1, frame_l.size(0), 1, frame_l.size(1), frame_l.size(2))
                t_input = frame_l if i == 0 else torch.cat((t_input, frame_l), 2)

            if nframes == 1:
                # add the same frame, because the number of frames for inference must be at least = 2
                t_input = torch.cat((t_input, frame_l), 2)

            output_l = t_input.to(self.device)

            out_frames: list[Image] = []

            if self.refimgs is None:
                output_ab = self.modelC(output_l)
            else:
                output_ab = self.modelC(output_l, self.refimgs)
            output_l = output_l.detach().cpu()
            output_ab = output_ab.detach().cpu()
            for i in range(nframes):
                out_l = output_l[0, :, i, :, :]
                out_c = output_ab[0, :, i, :, :]
                output = torch.cat((out_l, out_c), dim=0).numpy().transpose((1, 2, 0))
                if convert_to_pil:
                    output = Image.fromarray(np.uint8(convertLAB2RGB(output) * 255))
                else:
                    output = np.uint8(convertLAB2RGB(output) * 255)
                out_frames.append(output)

            torch.cuda.empty_cache()
            return out_frames
