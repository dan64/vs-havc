"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-06-21
version:
LastEditors: Dan64
LastEditTime: 2025-02-09
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
main Vapoursynth wrapper for DeepRemaster: Temporal Source-Reference Attention
Networks for Comprehensive Video Enhancement.
URL: https://github.com/satoshiiizuka/siggraphasia2019_remastering
"""

from __future__ import annotations
#import math
#import os
#from PIL import Image
#import numpy as np
import torch
from functools import partial

from vsdeoldify.vsslib.imfilters import image_weighted_merge
from vsdeoldify.vsslib.vsutils import debug_ModifyFrame
from vsdeoldify.vsslib.vsfilters import vs_tweak
from vsdeoldify.vsslib.vsscdect import BuildSCDetect

from vsdeoldify.vsslib.constants import *

from vsdeoldify.remaster.remaster_render import RemasterColorizer, RemasterEngine
from vsdeoldify.remaster.remaster_utils import *


os.environ["CUDA_MODULE_LOADING"] = "LAZY"

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model")


def vs_sc_remaster_colorize(
        clip: vs.VideoNode = None,
        clip_ref: vs.VideoNode = None,
        clip_sc: vs.VideoNode = None,
        length: int = 2,
        render_vivid: bool = False,
        ref_minedge: int = 256,
        frame_mindim: int = 320,
        ref_buffer_size: int = 10,
        ref_frequency: int = 0,
        device_index: int = 0,
        inference_mode: bool = False,
        merge_weight: float = 1.0,
        weights_dir: str = model_dir
) -> vs.VideoNode:
    """Function to perform colorization with DeepRemaster using Vapoursynth Video Frame to access the reference frames.

    :param clip:            Clip to process. Only RGB24 "full range" format is supported.
    :param clip_ref:        Clip with the reference frames, must be of the same size of input clip: Default: None
    :param clip_sc:         Clip with the scene change information: Default: None
    :param length:          Sequence length that the model processes (min. 2, max. 5). Default: 2
    :param render_vivid:    Given that the generated colors by the inference are a little washed out, by enabling
                            this parameter, the saturation will be increased by about 10%. range [True, False]
    :param ref_minedge:     min dimension of reference frames used for inference. Default: 256
    :param frame_mindim:    min dimension of input frames used for inference. Default: 320
    :param ref_buffer_size: reference frame buffer size for inference. Default: 100
    :param ref_frequency:   frequency used to set the reference frames. Default: 0 (not available)
    :param device_index:    Device ordinal of the GPU (if = -1 CPU mode is enabled). Default: 0
    :param inference_mode:  Enable/Disable torch inference mode. Default: False
    :param merge_weight:    Weight used to merge the reference frames. Default: 1.0 (no merge)
    :param weights_dir:     Path string of location of model weights (*.pth.tar).
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("HAVC_DeepRemaster: this is not a clip")

    if not isinstance(clip_ref, vs.VideoNode):
        raise vs.Error("HAVC_DeepRemaster: this is not a clip")

    if clip.format.id != vs.RGB24:
        raise vs.Error("HAVC_DeepRemaster: only RGB24 format is supported for input clip")

    if clip_ref.format.id != vs.RGB24:
        raise vs.Error("HAVC_DeepRemaster: only RGB24 format is supported for input clip_ref")

    if device_index != -1 and not torch.cuda.is_available():
        raise vs.Error("HAVC_DeepRemaster: CUDA is not available")

    if length < 2:
        raise vs.Error("HAVC_DeepRemaster: length must be at least 2")

    disable_warnings()

    if render_vivid:
        clip_ref = vs_tweak(clip_ref, hue=DEF_VIVID_HUE_LOW, sat=DEF_VIVID_SAT_HIGH)

    # enable torch inference mode
    # https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html
    if inference_mode:
        torch.backends.cudnn.benchmark = True
        torch.inference_mode()

    clip_orig = clip
    clip = resize_for_inference(clip, frame_mindim)
    engine = RemasterColorizer(clip_ref=clip_ref,
                               device_index=device_index,
                               ref_minedge=ref_minedge,
                               ref_buffer_size=ref_buffer_size,
                               model_dir=weights_dir)

    if clip_sc is None:
        clip_sc = BuildSCDetect(clip_ref)

    num_refs = engine.load_clip_ref(clip_sc)
    if num_refs == 0:
        raise vs.Error(f"HAVC_DeepRemaster: no reference frames found in clip_ref")

    base = clip.std.BlankClip(width=clip.width, height=clip.height, keep=True)

    # ----------------------------------------- INFERENCE -------------------------------------------------------------

    # create a local dictionary to store the colored frames
    local_cache: any = {}

    def color_remaster(n: int, f: list[vs.VideoFrame], v_clip: vs.VideoFrame = None,
                       colorizer: RemasterColorizer = None, batch_size: int = 2) -> vs.VideoFrame:

        if str(n) not in local_cache:
            local_cache.clear()

            frames = [frame_to_np_array(f[0])]

            for i in range(1, batch_size):
                if n + i >= v_clip.num_frames:
                    break
                frame_i = v_clip.get_frame(n + i)
                frames.append(frame_to_np_array(frame_i))

            last_frame_idx = min(n + batch_size - 1, v_clip.num_frames - 1)
            output = colorizer.process_frames(frames=frames, last_frame_idx=last_frame_idx)

            for i in range(len(output)):
                local_cache[str(n + i)] = output[i]

        np_frame = local_cache[str(n)]

        return np_array_to_frame(np_frame, f[1].copy())

    def color_merge_remaster(n: int, f: list[vs.VideoFrame], v_clip: vs.VideoFrame = None, mweight: float = 1.0,
                             colorizer: RemasterColorizer = None, batch_size: int = 2) -> vs.VideoFrame:

        if str(n) not in local_cache:
            local_cache.clear()

            frames = [frame_to_np_array(f[0])]

            for i in range(1, batch_size):
                if n + i >= v_clip.num_frames:
                    break
                frame_i = v_clip.get_frame(n + i)
                frames.append(frame_to_np_array(frame_i))

            last_frame_idx = min(n + batch_size - 1, v_clip.num_frames - 1)
            output = colorizer.process_frames(frames=frames, last_frame_idx=last_frame_idx, convert_to_pil=True)

            for i in range(len(output)):
                local_cache[str(n + i)] = output[i]

        f_img = local_cache[str(n)]
        ref_img = frame_to_image(f[2])
        if f_img.size != ref_img.size:
            ref_img = ref_img.resize(f_img.size, Image.Resampling.LANCZOS)
        f_merge = image_weighted_merge(f_img, ref_img, mweight)

        return image_to_frame(f_merge, f[1].copy())

    # ----------------------------------------- ModifyFrame -----------------------------------------------------------

    if 0 < merge_weight < 1 and not (clip_sc is None):
        clip_colored = base.std.ModifyFrame(clips=[clip, base, clip_ref], selector=partial(color_merge_remaster,
                                            v_clip=clip, mweight=merge_weight, colorizer=engine, batch_size=length))

        #clip_colored = debug_ModifyFrame(0, 720, base, clips=[clip, base, clip_ref],
        #                                 selector=partial(color_merge_remaster, v_clip=clip, mweight=merge_weight,
        #                                 colorizer=engine, batch_size=length), silent=False)

    else:
        clip_colored = base.std.ModifyFrame(clips=[clip, base, clip_ref], selector=partial(color_remaster, v_clip=clip,
                                            colorizer=engine, batch_size=length))

        #clip_colored = debug_ModifyFrame(0, 525, base, clips=[clip, base, clip_ref],
        #                                 selector=partial(color_remaster, v_clip=clip,
        #                                 colorizer=engine, batch_size=length), silent=False)

    clip_resized = clip_colored.resize.Spline64(width=clip_orig.width, height=clip_orig.height)

    clip_new = vs_recover_resolution(clip_orig, clip_resized)

    if render_vivid:
        clip_new = vs_tweak(clip_new, hue=DEF_VIVID_HUE_HIGH, sat=DEF_VIVID_SAT_LOW)

    return clip_new


def vs_remaster_colorize(
        clip: vs.VideoNode,
        length: int = 2,
        render_vivid: bool = False,
        ref_dir: str = None,
        ref_minedge: int = 256,
        frame_mindim: int = 320,
        ref_buffer_size: int = 20,
        device_index: int = 0,
        inference_mode: bool = False,
        weights_dir: str = model_dir
) -> vs.VideoNode:
    """Function to perform colorization with DeepRemaster using direct access to the reference frames folder

    :param clip:            Clip to process. Only RGB24 "full range" format is supported.
    :param length:          Sequence length that the model processes (min. 2, max. 5). Default: 2
    :param render_vivid:    Given that the generated colors by the inference are a little washed out, by enabling
                            this parameter, the saturation will be increased by about 10%. range [True, False]
    :param ref_dir:         Path of the reference frames. Default: None
    :param ref_minedge:     min dimension of reference frames used for inference. Default: 256
    :param frame_mindim:    min dimension of input frames used for inference. Default: 320
    :param ref_buffer_size: reference frame buffer size for inference. Default: 20
    :param device_index:    Device ordinal of the GPU (if = -1 CPU mode is enabled). Default: 0
    :param inference_mode:  Enable/Disable torch inference mode. Default: False
    :param weights_dir:     Path string of location of model weights (*.pth.tar).
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("HAVC_DeepRemaster: this is not a clip")

    if clip.format.id != vs.RGB24:
        raise vs.Error("HAVC_DeepRemaster: only RGB24 format is supported")

    if not os.path.isdir(ref_dir):
        raise vs.Error("HAVC_DeepRemaster: '", ref_dir, "' is not a valid directory")

    if device_index != -1 and not torch.cuda.is_available():
        raise vs.Error("HAVC_DeepRemaster: CUDA is not available")

    if length < 2:
        raise vs.Error("HAVC_DeepRemaster: length must be at least 2")

    disable_warnings()

    # enable torch inference mode
    # https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html
    if inference_mode:
        torch.backends.cudnn.benchmark = True
        torch.inference_mode()

    clip_orig = clip
    clip = resize_for_inference(clip, frame_mindim)
    engine = RemasterEngine(device_index=device_index,
                            ref_minedge=ref_minedge,
                            ref_buffer_size=ref_buffer_size,
                            model_dir=weights_dir)

    num_refs = engine.load_ref_dir(rf_dir=ref_dir)
    if num_refs == 0:
        raise vs.Error(f"HAVC_DeepRemaster: no reference frames found in {ref_dir}")

    base = clip.std.BlankClip(width=clip.width, height=clip.height, keep=True)

    # ----------------------------------------- INFERENCE -------------------------------------------------------------

    cache = {}

    def inference_remaster(n: int, f: list[vs.VideoFrame], v_clip: vs.VideoFrame = None, engine: RemasterEngine = None,
                           batch_size: int = 5) -> vs.VideoFrame:

        if str(n) not in cache:
            cache.clear()

            frames = [frame_to_np_array(f[0])]

            for i in range(1, batch_size):
                if n + i >= v_clip.num_frames:
                    break
                frame_i = v_clip.get_frame(n + i)
                frames.append(frame_to_np_array(frame_i))

            last_frame_idx = min(n + batch_size - 1, v_clip.num_frames - 1)
            output = engine.process_frames(frames=frames, last_frame_idx=last_frame_idx)

            for i in range(len(output)):
                cache[str(n + i)] = output[i]

        np_frame = cache[str(n)]

        return np_array_to_frame(np_frame, f[1].copy())

    # ----------------------------------------- ModifyFrame -----------------------------------------------------------

    clip_colored = base.std.ModifyFrame(clips=[clip, base], selector=partial(inference_remaster, v_clip=clip,
                                                                             engine=engine, batch_size=length))

    #clip_colored = debug_ModifyFrame(0, 35, base, clips=[clip, base], selector=partial(inference_remaster,
    #                                                        v_clip=clip, engine=engine, batch_size=length))

    clip_resized = clip_colored.resize.Spline64(width=clip_orig.width, height=clip_orig.height)

    clip_new = vs_recover_resolution(clip_orig, clip_resized)

    if render_vivid:
        clip_new = vs_tweak(clip_new, hue=DEF_VIVID_HUE_HIGH, sat=DEF_VIVID_SAT_LOW)

    return clip_new
