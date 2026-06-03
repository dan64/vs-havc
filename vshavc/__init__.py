"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-02-29
version: 
LastEditors: Dan64
LastEditTime: 2026-05-24
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
main Vapoursynth wrapper to pytorch-based coloring filter HybridAVC (HAVC).
The filter includes some portions of code from the following coloring projects:
DeOldify: https://github.com/jantic/DeOldify
DDColor: https://github.com/HolyWu/vs-ddcolor
Colorization: https://github.com/richzhang/colorization
Deep-Exemplar: https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization
ColorMNet: https://github.com/yyang181/colormnet
CMNET2: https://github.com/dan64/cmnet2
Deep-Remaster: https://github.com/satoshiiizuka/siggraphasia2019_remastering
"""
from __future__ import annotations

import os

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_LOGS"] = "-all"

import pathlib
import math

from vshavc.havc_utils import _get_tune_id, convert_format_RGB24, restore_format, HAVC_read_video, rgb_denoise
from vshavc.havc_utils import  rgb_balance, rgb_equalizer, vs_auto_levels
from vshavc.vsslib.mcomb import vs_sc_combine_models, vs_combine_models, vs_ext_reference_clip, ChromaRetentionMerge
from vshavc.vsslib.vsfilters import vs_rgb_normalize, vs_simple_merge, vs_tweak, vs_sc_colormap, vs_sc_dark_tweak
from vshavc.vsslib.vsfilters import  vs_sc_chroma_bright_tweak, vs_sc_recover_clip_luma, vs_recover_clip_luma
from vshavc.vsslib.vsfilters import vs_dark_tweak, vs_chroma_bright_tweak, vs_colormap, vs_chroma_stabilizer_ex
from vshavc.vsslib.vsfilters import vs_get_clip_frame
from vshavc.vsslib.vsmodels import vs_sc_deoldify, vs_sc_ddcolor, vs_colormnet, vs_deepex, vs_deepremaster
from vshavc.vsslib.vsmodels import vs_colormnet2, vs_colormnet2dit
from vshavc.vsslib.vsplugins import vs_reduce_flicker, vs_timecube
from vshavc.vsslib.vsretinex import vs_retinex
from vshavc.vsslib.vsutils import vs_sc_export_frames, vs_list_export_frames, HAVC_LogMessage, MessageType
from vshavc.vsslib.vsutils import frame_to_image
from vshavc.vsslib.vsresize import SmartResizeColorizer, SmartResizeReference, get_render_size
from vshavc.vsslib.vsscdect import SceneDetectFromDir, SceneDetect, CopySCDetect
from vshavc.vsslib.vsscdect import get_sc_props, vs_mv_sc_detect, vs_sc_xvid
from vshavc.vsslib.vsscdetect_edge import SceneDetectEdges
from vshavc.vsslib.vstiles4 import ClipTiles
from vshavc.vsslib.vstiles4 import vs_slice_into_2_horizontal_tiles, vs_reconstruct_from_2_horizontal_tiles
from vshavc.vsslib.vstiles4 import vs_slice_into_2x2_overlapping_tiles, vs_reconstruct_from_2x2_overlapping_tiles

from vshavc.deoldify import device
from vshavc.deoldify.device_id import DeviceId
from vshavc.deepex import deepex_colorizer, get_deepex_size, ModelColorizer
import vshavc.remaster

import vshavc.vsslib.constants as constants

__version__ = "5.8.5"

import warnings
import logging

from typing import Sequence

warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the "
                                "future, please use 'weights' instead.")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None`.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.nn.utils.weight_norm is deprecated.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="Conversion from CIE-LAB,*?")
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Torch was not compiled with flash attention.*?")

warnings.filterwarnings("ignore", category=FutureWarning, message=".torch.cuda.amp.custom_fwd.*?")
warnings.filterwarnings("ignore", category=FutureWarning, message="Arguments other than a weight enum or `None`.*?")
warnings.filterwarnings("ignore", category=FutureWarning, message=".You are using `torch.load`.*?")

warnings.simplefilter(action='ignore', category=FutureWarning)

package_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(package_dir, "models")

# configuring torch
import torch
torch.backends.cudnn.benchmark = True

import vapoursynth as vs

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to HAVC filter with "presets" management
"""

def HAVC_main(clip: vs.VideoNode, Preset: str = 'Medium', FrameInterp: int = 0,  ColorModel: str = 'Video+ModelScope',
              CombMethod: str = 'Simple',  VideoTune: str = 'VeryVivid', ColorFix: str = 'Retinex/Red',
              ColorTune: str = 'Light', ColorMap: str = 'Red->Brown', ColorTemp: str = "None", BlackWhiteTune: str = 'Light',
              BlackWhiteMode: int = 0, BlackWhiteBlend: bool = True, EnableDeepEx: bool = True, DeepExMethod: int = 0,
              DeepExPreset: str = 'Auto', DeepExRefMerge: int = 0, DeepExOnlyRefFrames: bool = False,
              ScFrameDir: str = None, ScThreshold: float = constants.DEF_THRESHOLD, ScThtOffset: int = 1, ScMinFreq: int = 15,
              ScMinInt: int = 1, ScThtSSIM: float = 0.0, ScNormalize: bool = False, DeepExModel: int = 0,
              DeepExVivid: bool = False, DeepExEncMode: int = 0, DeepExMaxMemFrames=20, RefRange: tuple[int, int] = (0, 0),
              enable_fp16: bool = True, debug_level: int = 0) -> vs.VideoNode:
    """Main HAVC function supporting the Presets

    :param clip:                clip to process, any format is supported.
    :param Preset:              Preset to control the encoding speed/quality.
                                Allowed values are:
                                    'Placebo', (HD coloring support, speed-up with FrameInterp)
                                    'VerySlow',
                                    'Slower',
                                    'Slow',
                                    'Medium', (default)
                                    'Fast',
                                    'Faster',
                                    'VeryFast'
    :param FrameInterp:         This parameter will allow to enable the frame interpolation. This method will use
                                Deep-Exemplar to interpolate the colored frames if FrameInterp < 5, otherwise will use
                                ColorMNet. If = 0, the interpolation is disabled, if > 0 represent the number of frames
                                used for interpolation. The quality of interpolation will decrease with the number of
                                frames, suggested value is 3. Range [0-10], Default = 0
    :param ColorModel:          Preset to control the Color Models to be used for the color inference
                                Allowed values are:
                                    'Video+Artistic'
                                    'Stable+Artistic'
                                    'Video+ModelScope' (default)
                                    'Stable+ModelScope'
                                    'Artistic+Modelscope'
                                    'Video+Siggraph17'
                                    'Video+ECCV16'
                                    'DeOldify(Video)'
                                    'DeOldify(Stable)'
                                    'DeOldify(Artistic)'
                                    'DDColor(Artistic)'
                                    'DDColor(ModelScope)'
                                    'Zhang(Siggraph17)'
                                    'Zhang(ECCV16)'
    :param CombMethod:          Method used to combine coloring models with (+):
                                Allowed values are:
                                    'Simple' (default)
                                    'Constrained-Chroma'
                                    'Luma-Masked'
                                    'Adaptive-Luma'
                                    'Chroma-Retention'
                                    'ChromaBound Adaptive'
    :param VideoTune:           Preset to control the output video color stability
                                Allowed values are:
                                    'VeryStable',
                                    'MoreStable'
                                    'Stable',
                                    'Balanced',
                                    'Vivid',
                                    'MoreVivid',
                                    'VeryVivid',  (default)
    :param ColorFix:            This parameter allows to reduce color noise on specific chroma ranges.
                                Allowed values are:
                                    'None',
                                    'Retinex/Red'   (default)
                                    'Magenta',
                                    'Magenta/Violet',
                                    'Violet',
                                    'Violet/Red',
                                    'Blue/Magenta',
                                    'Yellow',
                                    'Yellow/Orange',
                                    'Yellow/Green'
    :param ColorTune:           This parameter allows to define the intensity of noise reduction applied by ColorFix.
                                Allowed values are:
                                    'None'
                                    'Light',  (default)
                                    'Medium',  
                                    'Strong',
    :param ColorMap:            This parameter allows to change a given color range to another color.
                                Allowed values are:
                                    'None',
                                    'Blue->Brown',
                                    'Blue->Red',
                                    'Blue->Green',
                                    'Green->Brown',
                                    'Green->Red',
                                    'Green->Blue',
                                    'Red->Brown',  (default)
                                    'Red->Blue'
                                    'Yellow->Rose'
    :param ColorTemp:           Strength of the color temporal stabilization filter. This post process filter will be
                                applied only to the DDColor family color models: DDColor(Artistic), DDColor(ModelScope),
                                Zhang(Siggraph17), Zhang(ECCV16). Allowed values are:
                                        "None" (default)
                                        "Very High"
                                        "High"
                                        "Medium"
                                        "Low"
                                        "Very Low"
    :param BlackWhiteTune:      This parameter allows to improve contrast and luminosity of frames colored with HAVC.
                                Allowed values are:
                                    'None'
                                    'Light',  (default)
                                    'Medium',
                                    'Strong'
    :param BlackWhiteMode:      Method used by BlackWhiteTune to perform colors adjustments.
                                Allowed values are:
                                        0 : CLAHE (luma) (default)
                                        1 : Simple (RGB)
                                        2 : CLAHE (RGB)
                                        3 : CLAHE (luma) + Simple (RGB)
                                        4 : ScaleAbs – LUT
                                        5 : Multi-Scale Retinex (HAVC)
                                        6 : Multi-Scale Retinex (B&W)
    :param BlackWhiteBlend:     If enabled the frames adjusted with BlackWhiteTune will be blended with the original frames.
                                Default = True 
    :param EnableDeepEx:        Enable coloring using "Exemplar-based" Video Colorization models.
                                Default = True
    :param DeepExMethod:        Method to use to generate reference frames.
                                        0 = HAVC same as video (default)
                                        1 = HAVC + RF same as video
                                        2 = HAVC + RF different from video
                                        3 = external RF same as video
                                        4 = external RF different from video
                                        5 = external ClipRef same as video
                                        6 = external ClipRef different from video
    :param DeepExPreset:        Preset to control the render method and speed:
                                Allowed values are:
                                        'Auto'   : will be automatically assigned the optimal render size (default)
                                        'Fast'   : colors are more washed out
                                        'Medium' : colors are a little washed out
                                        'Slow'   : colors are a little more vivid
                                        'Slower' : colors are more accurate (usually is very slow)
    :param DeepExRefMerge:      Method used by DeepEx to merge the reference frames with the frames propagated by DeepEx.
                                It is applicable only with DeepEx method: 0, 1, 2.
                                Allowed values are:
                                        0 = No RF merge (reference frames can be produced with any frequency) (default)
                                        1 = RF-Merge VeryLow (reference frames are merged with weight=0.3)
                                        2 = RF-Merge Low (reference frames are merged with weight=0.4)
                                        3 = RF-Merge Med (reference frames are merged with weight=0.5)
                                        4 = RF-Merge High (reference frames are merged with weight=0.6)
                                        5 = RF-Merge VeryHigh (reference frames are merged with weight=0.7)
    :param DeepExOnlyRefFrames: If enabled the filter will output in "ScFrameDir" the reference frames. Useful to check
                                and eventually correct the frames with wrong colors (can be used only if DeepExMethod = 0).
                                Default = False
    :param DeepExModel:         Exemplar Model used by DeepEx to propagate color frames.
                                        0 : CMNET2 (default)
                                        1 : Deep-Exemplar
                                        2 : Deep-Remaster
                                        3 : ColorMNet
    :param DeepExVivid:         Depending on selected DeepExModel, if enabled:
                                    0) CMNET2: the saturation will be increased by about 15%.
                                    1) Deep-Exemplar: the saturation will be increased by about 25%.
                                    2) Deep-Remaster: the saturation will be increased by about 20% and Hue by +10.
                                    3) ColorMNet: the frames memory is reset at every reference frame update
                                range [True, False]. Default = False
    :param DeepExEncMode:       Parameter used by CMNET2/ColorMNet to define the encode mode strategy.
                                Available values are:
                                     0: remote encoding. The frames will be colored by a thread outside Vapoursynth.
                                                         This option don't have any GPU memory limitation and will allow
                                                         to fully use the long term frame memory.
                                                         It is the faster encode method (default)
                                     1: local encoding.  The frames will be colored inside the Vapoursynth environment.
                                                         In this case the max_memory will be limited by the size of GPU
                                                         memory (max 15 frames for 24GB GPU).
                                                         Useful for coloring clips with a lot of smooth transitions,
                                                         since in this case is better to use a short frame memory or
                                                         the Deep-Exemplar model, which is faster.
    :param DeepExMaxMemFrames:  Parameter used by CMNET2/ColorMNet/DeepRemaster models.
                                For ColorMNet specify the max number of encoded frames to keep in memory. Default = 0
                                Its value depend on encode mode and must be defined manually following the suggested values.
                                DeepExEncMode=0: there is no memory limit (it could be all the frames in the clip).
                                Suggested values are:
                                    min=50, max=5000
                                If = 0 will be filled with the value of 5000 or the clip length if lower.
                                DeepExEncMode=1: the max memory frames is limited by available GPU memory.
                                Suggested values are:
                                    min=1, max=2      : for 8GB GPU
                                    min=1, max=4      : for 12GB GPU
                                    min=1, max=6      : for 16GB GPU
                                If = 0 will be filled with the max value (depending on total GPU RAM available)
                                For CMNET2/DeepRemaster represent the number to reference frames to keep in memory.
                                Suggested values are:
                                    min=4, max=50 (100 CMNET2)
                                If = 0 will be filled with the value of 20. Default = 20
    :param ScFrameDir:          if set, define the directory where are stored the reference frames that will be used
                                by "Exemplar-based" Video Colorization models. With DeepExMethod 5,6 this parameter
                                can be the path to a video clip. Default = None
    :param ScThreshold:         Scene change threshold used to generate the reference frames to be used by
                                "Exemplar-based" Video Colorization. It is a percentage of the luma change between
                                the previous and the current frame. range [0-1], default 0.10. If =0 are not generate
                                reference frames.
    :param ScThtOffset:         Offset index used for the Scene change detection. The comparison will be performed,
                                between frame[n] and frame[n-offset]. An offset > 1 is useful to detect blended scene
                                change, range[1, 25]. Default = 1.
    :param ScMinInt:            Minimum number of frame interval between scene changes, range[1, 25]. Default = 1.
    :param ScMinFreq:           if > 0 will be generated at least a reference frame every "ScMinFreq" frames.
                                range [0-1500], default: 15.
    :param ScThtSSIM:           Threshold used by the SSIM (Structural Similarity Index Metric) selection filter.
                                If > 0, will be activated a filter that will improve the scene-change detection,
                                by discarding images that are similar.
                                Suggested values are between 0.35 and 0.65, range [0-1], default 0.0 (deactivated)
    :param ScNormalize:         If true the B&W frames are normalized before use misc.SCDetect(), the normalization will
                                increase the sensitivity to smooth scene changes, range [True, False], default: False
    :param RefRange:            Parameter used only with DeepExMethod in (5, 6). With this parameter it is possible to
                                provide the frame number of clip start and end. For example RefRange=(100, 500)
                                will return the clip's slice: clip[100:500], if RefRange=(0, 0) will be considered all
                                clip's frames.
    :param enable_fp16:         Enable/disable FP16 in ddcolor inference, range [True, False]. Default = True
    :param debug_level:         Set the level of HAVC debug messages. Default = 0 (no messages)
    """
    # disable packages warnings
    disable_warnings()

    HAVC_set_debug_level(debug_level)

    # Select presets / tuning
    # speed_id, deoldify_rf, ddcolor_rf = havc_utils._get_render_factors(Preset)
    
    return HAVC_main_presets(clip, Preset, FrameInterp, ColorModel, CombMethod, VideoTune, ColorFix,
               ColorTune, ColorMap, ColorTemp, BlackWhiteTune, BlackWhiteMode, BlackWhiteBlend, EnableDeepEx,
               DeepExMethod, DeepExPreset, DeepExRefMerge, DeepExOnlyRefFrames, ScFrameDir, ScThreshold, ScThtOffset,
               ScMinFreq, ScMinInt, ScThtSSIM, ScNormalize, DeepExModel, DeepExVivid, DeepExEncMode, DeepExMaxMemFrames,
               RefRange, enable_fp16, debug_level)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Main HAVC function supporting Presets
"""

def HAVC_veryslow_preset(clip: vs.VideoNode, Preset: str = 'Slower', FrameInterp: int = 0,
                         ColorModel: str = 'Video+Artistic', CombMethod: str = 'Simple', VideoTune: str = 'Stable',
                         ColorFix: str = 'Magenta/Violet', ColorTune: str = 'Light', ColorMap: str = 'None',
                         ColorTemp: str = "None", BlackWhiteTune: str = 'None', BlackWhiteMode: int = 0,
                         BlackWhiteBlend: bool = True, EnableDeepEx: bool = False, DeepExMethod: int = 0,
                         ScThreshold: float = 0.1, ScMinFreq: int = 0, RefRange: tuple[int, int] = (0, 0),
                         enable_fp16: bool = True, debug_level: int = 0) -> vs.VideoNode:
    """
    HAVC function supporting Preset VerySlow: Deep-Exemplar settings is not supported, only ColorTemp and FrameInterp
    """

    def veryslow_preset(clip: vs.VideoNode, Preset: str = 'Slower', FrameInterp: int = 0,
                        ColorModel: str = 'Video+Artistic', CombMethod: str = 'Simple', VideoTune: str = 'Stable',
                        ColorFix: str = 'Magenta/Violet', ColorTune: str = 'Light', ColorMap: str = 'None',
                        ColorTemp: str = "None", BlackWhiteTune: str = 'None', BlackWhiteMode: int = 0,
                        BlackWhiteBlend: bool = True, EnableDeepEx: bool = False, DeepExMethod: int = 0,
                        ScThreshold: float = 0.1, ScMinFreq: int = 0, RefRange: tuple[int, int] = (0, 0),
                        enable_fp16: bool = True, debug_level: int = 0) -> vs.VideoNode:
        """Internal colorization pipeline: runs DeOldify and DDColor separately,
        applies per-model colour and contrast tweaks, then merges the two results.

        This nested function is called by HAVC_veryslow_preset to build the base colourised
        clip before any temporal stabilisation (ColorTemp) or frame interpolation is applied.
        """
        deoldify_model, ddcolor_model = havc_utils._spit_color_model(ColorModel)
        # -- Clip1: DeOldify --------------------------
        if deoldify_model != "none":
            clip_dark = HAVC_tweak(clip=clip, bright=-1, gamma=0.90, cont=0.80)
            clip1 = HAVC_main_presets(clip=clip_dark, Preset=Preset, ColorModel=deoldify_model, ColorTemp="none", ColorFix="none",
                                      ColorTune="medium", BlackWhiteTune="light", BlackWhiteMode=0, BlackWhiteBlend=True,
                                      FrameInterp=0, EnableDeepEx=EnableDeepEx, DeepExMethod=DeepExMethod,
                                      ScThreshold=ScThreshold, ScMinFreq=ScMinFreq, RefRange=RefRange,
                                      enable_fp16=enable_fp16, debug_level=debug_level)
            clip1 = HAVC_ColorAdjust(clip1, BlackWhiteTune='medium', BlackWhiteMode=4, BlackWhiteBlend=True, ReColor=False,
                                     chroma_resize=True)
            clip1 = HAVC_tweak(clip=clip1, sat=0.95, hue=5)
        else:
            clip1 = None
        # -- Clip2: DDColor -------------------------- Support: FrameInterp
        if ddcolor_model != "none":
            clip_dark = HAVC_tweak(clip=clip, bright=-1, gamma=0.95, cont=0.95)
            clip2 = HAVC_main_presets(clip=clip_dark, Preset=Preset, ColorModel=ddcolor_model, ColorTemp="none",
                                  ColorFix=ColorFix, ColorMap=ColorMap, ColorTune=ColorTune, BlackWhiteMode=0,
                                  FrameInterp=0, BlackWhiteTune="light", BlackWhiteBlend=True,
                                  EnableDeepEx=EnableDeepEx, DeepExMethod=DeepExMethod, ScThreshold=ScThreshold,
                                  ScMinFreq=ScMinFreq, RefRange=RefRange, enable_fp16=enable_fp16, debug_level=debug_level)
        else:
            clip2 = None
        # -------------------------------
        if clip1 is None:
            clip_color = HAVC_merge(clipa=clip2, clip_luma=clip, method=0)
        elif clip2 is None:
            clip_color = HAVC_merge(clipa=clip1, clip_luma=clip, method=0)
        else:
            ddcolor_weight = havc_utils._get_mweight(VideoTune)
            dd_method = havc_utils._get_comb_method(CombMethod)
            clip_color = HAVC_merge(clipa=clip1, clipb=clip2, clip_luma=clip, weight=ddcolor_weight, method=dd_method)
        return clip_color
    # -----------------------------------------------------------------------------------------------------------
    # Start code for HAVC_veryslow_preset()
    # -----------------------------------------------------------------------------------------------------------
    clip, orig_fmt = convert_format_RGB24(clip, chroma_resize=False)

    if FrameInterp == 0:  # Very Slow encoding (2x slower)
        color_temp = havc_utils._get_temp_color(ColorTemp)
        clip_colored = veryslow_preset(clip, "Slower", 0, ColorModel, CombMethod, VideoTune, ColorFix,
                                         ColorTune, ColorMap, "None", BlackWhiteTune, BlackWhiteMode, BlackWhiteBlend,
                                         RefRange=RefRange, enable_fp16=enable_fp16, debug_level=debug_level)
        if color_temp > 0:
            clip_ref = clip_colored.std.SetFrameProp(prop="sc_threshold", floatval=0.1)
            clip_ref = clip_ref.std.SetFrameProp(prop="sc_frequency", intval=1)
            clip_colored = HAVC_cmnet2(clip=clip, clip_ref=clip_ref, render_speed='Medium', render_vivid=False,
                                     ref_merge=color_temp, dark=True, dark_p=[0.2, 0.8], ref_thresh=0.10,
                                     encode_mode=0, max_memory_frames=10, ref_freq=0, ref_norm=True,
                                     smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"], colormap="300:360|0.8,0.1")

    else:  # Faster Encoding, using ColorMNet as frame interpolator
        ref_freq_temp = FrameInterp if FrameInterp < 5 else FrameInterp * 2
        clip_colored = veryslow_preset(clip, "Slower", 0, ColorModel, CombMethod, VideoTune, ColorFix,
                                         ColorTune, ColorMap, "None", BlackWhiteTune, BlackWhiteMode, BlackWhiteBlend,
                                         EnableDeepEx=True, DeepExMethod=constants.DEF_HAVC_METHOD_PLACEBO,
                                         ScThreshold=0.1,  ScMinFreq=ref_freq_temp, RefRange=RefRange,
                                         enable_fp16=enable_fp16, debug_level=debug_level)
        clip_ref = clip_colored.std.SetFrameProp(prop="sc_threshold", floatval=0.1)
        clip_ref = clip_ref.std.SetFrameProp(prop="sc_frequency", intval=ref_freq_temp)
        clip_colored = vs_frame_interpolation(clip=clip, clip_ref=clip_ref, frame_interp=FrameInterp,
                                              chroma_adjust="300:360|0.8,0.1", process_id=2)

    clip_adjusted = HAVC_ColorAdjust(clip_colored,  BlackWhiteTune=BlackWhiteTune, BlackWhiteMode=BlackWhiteMode,
                                     BlackWhiteBlend=BlackWhiteBlend, ReColor=False)

    clip_adjusted = HAVC_tweak(clip_adjusted, hue=10, sat=1.05, cont=0.90)
    clip_colored = HAVC_merge(clipa=clip_adjusted, clipb=clip_colored, weight=0.4, method=2)

    return restore_format(clip_colored, orig_fmt)

def HAVC_placebo_preset(clip: vs.VideoNode, CombMethod: str = 'Simple', VideoTune: str = 'Stable',
                        ColorModel: str = 'Video+Artistic', ColorFix: str = 'Magenta/Violet', ColorTune: str = 'Light',
                        ColorMap: str = 'None', ColorTemp: str = 'None', FrameInterp: int = 0,
                        BlackWhiteTune: str = 'None', BlackWhiteMode: int = 0, BlackWhiteBlend: bool = True,
                        RefRange: tuple[int, int] = (0, 0), enable_fp16: bool = True,
                        debug_level: int = 0) -> vs.VideoNode:
    """
    HAVC function supporting Preset Placebo: Deep-Exemplar settings is not supported, only ColorTemp and FrameInterp
    """
    clip, orig_fmt = convert_format_RGB24(clip, chroma_resize=False)

    if FrameInterp == 0:   # Very Slow encoding (4x slower)
        color_temp = havc_utils._get_temp_color(ColorTemp)
        clip_colored = HAVC_main_presets(clip, "Placebo", 0, ColorModel, CombMethod, VideoTune, ColorFix,
                       ColorTune, ColorMap, "None", BlackWhiteTune, BlackWhiteMode, BlackWhiteBlend,
                       RefRange=RefRange, enable_fp16=enable_fp16, debug_level=debug_level)
        if color_temp > 0:
            clip_ref = clip_colored.std.SetFrameProp(prop="sc_threshold", floatval=0.1)
            clip_ref = clip_ref.std.SetFrameProp(prop="sc_frequency", intval=1)
            clip_colored = HAVC_cmnet2(clip=clip, clip_ref=clip_ref, render_speed='Medium', render_vivid=False,
                                     ref_merge=color_temp, dark=True, dark_p=[0.2, 0.8], ref_thresh=0.10,
                                     encode_mode=0, max_memory_frames=1, ref_freq=0, ref_norm=True,
                                     smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"], colormap="300:360|0.8,0.1")

    else:  # Faster Encoding, using ColorMNet as frame interpolator
        ref_freq_temp = FrameInterp if FrameInterp < 5 else FrameInterp*2

        clip_colored = HAVC_main_presets(clip, "Placebo", 0, ColorModel, CombMethod, VideoTune, ColorFix,
                       ColorTune, ColorMap, "None", BlackWhiteTune, BlackWhiteMode, BlackWhiteBlend,
                       EnableDeepEx = True, DeepExMethod = constants.DEF_HAVC_METHOD_PLACEBO, ScThreshold = 0.1,
                       ScMinFreq = ref_freq_temp, RefRange=RefRange, enable_fp16=enable_fp16, debug_level=debug_level)

        clip_ref = clip_colored.std.SetFrameProp(prop="sc_threshold", floatval=0.1)
        clip_ref = clip_ref.std.SetFrameProp(prop="sc_frequency", intval=ref_freq_temp)
        clip_colored = vs_frame_interpolation(clip=clip, clip_ref=clip_ref, frame_interp=FrameInterp,
                                              chroma_adjust="300:360|0.8,0.1", process_id=1)

    return restore_format(clip_colored, orig_fmt)

def HAVC_main_presets(clip: vs.VideoNode, Preset: str = 'Medium', FrameInterp: int = 0, ColorModel: str = 'Video+Artistic',
                  CombMethod: str = 'Simple', VideoTune: str = 'Stable', ColorFix: str = 'Magenta/Violet',
                  ColorTune: str = 'Light', ColorMap: str = 'None', ColorTemp: str = "None",
                  BlackWhiteTune: str = 'None', BlackWhiteMode: int = 0, BlackWhiteBlend: bool = True,
                  EnableDeepEx: bool = False, DeepExMethod: int = 0, DeepExPreset: str = 'Medium',
                  DeepExRefMerge: int = 0, DeepExOnlyRefFrames: bool = False, ScFrameDir: str = None,
                  ScThreshold: float = constants.DEF_THRESHOLD, ScThtOffset: int = 1, ScMinFreq: int = 0,
                  ScMinInt: int = 1, ScThtSSIM: float = 0.0, ScNormalize: bool = False, DeepExModel: int = 0,
                  DeepExVivid: bool = False, DeepExEncMode: int = 0, DeepExMaxMemFrames=0,
                  RefRange: tuple[int, int] = (0, 0),
                  enable_fp16: bool = True, debug_level: int = 0) -> vs.VideoNode:
    """
    HAVC function supporting Presets: 'Slower', 'Slow', 'Medium', 'Fast', 'Faster', 'VeryFast'
    """

    # disable packages warnings
    disable_warnings()

    HAVC_set_debug_level(debug_level)

    # Select presets / tuning
    speed_id, deoldify_rf, ddcolor_rf = havc_utils._get_render_factors(Preset)

    chroma_resize: bool = (speed_id > constants.DEF_MIN_CPUID_RESIZE)   # 'placebo' and 'veryslow' will not be downsized

    clip, orig_fmt = convert_format_RGB24(clip, chroma_resize=chroma_resize)

    EnableRetinex: bool = ColorTune.lower() != "none" and ColorFix.lower() == "retinex/red"
    BWTuneRetinex: bool = BlackWhiteTune.lower() != "none" and BlackWhiteMode == 6
    DeFlicker: bool = EnableDeepEx or ColorTemp.lower() != "none" or EnableRetinex or BWTuneRetinex

    if BWTuneRetinex:
        clip = HAVC_bw_tune(clip, bw_tune=BlackWhiteTune, bw_method=5, luma_blend=BlackWhiteBlend)
        BlackWhiteTune = "light"
        BlackWhiteMode = 0
        BlackWhiteBlend = True
    
    clip_colored = HAVC_main_colorizer(clip, Preset, ColorModel, CombMethod,  VideoTune, ColorFix, ColorTemp,
              ColorTune, ColorMap, EnableDeepEx, DeepExMethod, DeepExPreset, DeepExRefMerge, DeepExOnlyRefFrames,
              ScFrameDir, ScThreshold, ScThtOffset, ScMinFreq, ScMinInt, ScThtSSIM, ScNormalize, DeepExModel,
              DeepExVivid, DeepExEncMode, DeepExMaxMemFrames, FrameInterp, RefRange, enable_fp16, debug_level)

    if BWTuneRetinex:
        clip_colored = HAVC_tweak(clip_colored, hue=5.0, sat=0.95, bright=0, cont=0.98, gamma=0.98)

    if BlackWhiteTune.lower() != "none":
        clip_colored = HAVC_bw_tune(clip_colored, BlackWhiteTune, BlackWhiteMode, BlackWhiteBlend)

    clip_final = clip_colored
    if EnableRetinex:
        match ColorTune.lower():
            case 'light':
                clip_final = vs_timecube(clip_colored, 0.8, constants.DEF_LUT_Exploration)
            case 'medium':
                clip_final = vs_timecube(clip_colored, 0.6, constants.DEF_LUT_City_Skyline)
            case 'strong':
                if ColorMap.lower() == "red->brown":
                    clip_final = vs_timecube(clip_colored, 0.4, constants.DEF_LUT_Amber_Light)
                else:
                    clip_final = vs_timecube(clip_colored, 0.6, constants.DEF_LUT_FUJ_Film)

    if DeFlicker:
        clip_final = vs_reduce_flicker(clip_final)

    return restore_format(clip_final, orig_fmt)

def HAVC_main_colorizer(clip: vs.VideoNode, Preset: str = 'Medium', ColorModel: str = 'Video+Artistic',
              CombMethod: str = 'Simple',  VideoTune: str = 'Stable', ColorFix: str = 'Magenta/Violet',
              ColorTemp: str = "None", ColorTune: str = 'Medium', ColorMap: str = 'None',  EnableDeepEx: bool = False,
              DeepExMethod: int = 0, DeepExPreset: str = 'Medium', DeepExRefMerge: int = 0,
              DeepExOnlyRefFrames: bool = False, ScFrameDir: str = None, ScThreshold: float = constants.DEF_THRESHOLD,
              ScThtOffset: int = 1, ScMinFreq: int = 0, ScMinInt: int = 1, ScThtSSIM: float = 0.0,
              ScNormalize: bool = False, DeepExModel: int = 0, DeepExVivid: bool = False, DeepExEncMode: int = 0,
              DeepExMaxMemFrames=0, FrameInterp: int = 0, RefRange: tuple[int, int] = (0, 0), enable_fp16: bool = True,
              debug_level: int = 0) -> vs.VideoNode:
    """Main HAVC function supporting the Presets

    :param clip:                clip to process, any format is supported.
    :param Preset:              Preset to control the encoding speed/quality.
                                Allowed values are:
                                    'Placebo', (HD coloring support, speed-up with FrameInterp)
                                    'VerySlow',
                                    'Slower',
                                    'Slow',
                                    'Medium', (default)
                                    'Fast',
                                    'Faster',
                                    'VeryFast'
    :param FrameInterp:         This parameter will allow to enable the frame interpolation. This method will use
                                Deep-Exemplar to interpolate the colored frames if FrameInterp < 5, otherwise will use
                                ColorMNet. If = 0, the interpolation is disabled, if > 0 represent the number of frames
                                used for interpolation. The quality of interpolation will decrease with the number of
                                frames, suggested value is 3. Range [0-10], Default = 0
    :param ColorModel:          Preset to control the Color Models to be used for the color inference
                                Allowed values are:
                                    'Video+Artistic'  (default)
                                    'Stable+Artistic'
                                    'Video+ModelScope'
                                    'Stable+ModelScope'
                                    'Artistic+Modelscope'
                                    'Video+Siggraph17'
                                    'Video+ECCV16'
                                    'DeOldify(Video)'
                                    'DeOldify(Stable)'
                                    'DeOldify(Artistic)'
                                    'DDColor(Artistic)'
                                    'DDColor(ModelScope)'
                                    'Zhang(Siggraph17)'
                                    'Zhang(ECCV16)'
    :param CombMethod:          Method used to combine coloring models with (+):
                                Allowed values are:
                                    'Simple' (default)
                                    'Constrained-Chroma'
                                    'Luma-Masked'
                                    'Adaptive-Luma'
                                    'Chroma-Retention'
                                    'ChromaBound Adaptive'
    :param VideoTune:           Preset to control the output video color stability
                                Allowed values are:
                                    'VeryStable',
                                    'MoreStable'
                                    'Stable',  (default)
                                    'Balanced',
                                    'Vivid',
                                    'MoreVivid',
                                    'VeryVivid',
    :param ColorFix:            This parameter allows to reduce color noise on specific chroma ranges.
                                Allowed values are:
                                    'None',
                                    'Retinex/Red'
                                    'Magenta',
                                    'Magenta/Violet',   (default)
                                    'Violet',
                                    'Violet/Red',
                                    'Blue/Magenta',
                                    'Yellow',
                                    'Yellow/Orange',
                                    'Yellow/Green'
    :param ColorTune:           This parameter allows to define the intensity of noise reduction applied by ColorFix.
                                Allowed values are:
                                    'None'
                                    'Light',
                                    'Medium',  (default)
                                    'Strong',
    :param ColorMap:            This parameter allows to change a given color range to another color.
                                Allowed values are:
                                    'None', (default)
                                    'Blue->Brown',
                                    'Blue->Red',
                                    'Blue->Green',
                                    'Green->Brown',
                                    'Green->Red',
                                    'Green->Blue',
                                    'Red->Brown',
                                    'Red->Blue'
                                    'Yellow->Rose'
    :param ColorTemp:           Strength of the color temporal stabilization filter. This post process filter will be
                                applied only to the DDColor family color models: DDColor(Artistic), DDColor(ModelScope),
                                Zhang(Siggraph17), Zhang(ECCV16). Allowed values are:
                                        "None" (default)
                                        "Very High"
                                        "High"
                                        "Medium"
                                        "Low"
                                        "Very Low"
    :param BlackWhiteTune:      This parameter allows to improve contrast and luminosity of frames colored with HAVC.
                                Allowed values are:
                                    'None' (default)
                                    'Light',
                                    'Medium',
                                    'Strong'
    :param BlackWhiteMode:      Method used by BlackWhiteTune to perform colors adjustments.
                                Allowed values are:
                                        0 : CLAHE (luma) (default)
                                        1 : Simple (RGB)
                                        2 : CLAHE (RGB)
                                        3 : CLAHE (luma) + Simple (RGB)
                                        4 : ScaleAbs – LUT
                                        5 : Multi-Scale Retinex (HAVC)
                                        6 : Multi-Scale Retinex (B&W)
    :param BlackWhiteBlend:     If enabled the frames adjusted with BlackWhiteTune will be blended with the original frames.
                                Default = True
    :param EnableDeepEx:        Enable coloring using "Exemplar-based" Video Colorization models.
                                Default = False
    :param DeepExMethod:        Method to use to generate reference frames.
                                        0 = HAVC same as video (default)
                                        1 = HAVC + RF same as video
                                        2 = HAVC + RF different from video
                                        3 = external RF same as video
                                        4 = external RF different from video
                                        5 = external ClipRef same as video
                                        6 = external ClipRef different from video
    :param DeepExPreset:        Preset to control the render method and speed:
                                Allowed values are:
                                        'Auto'   : will be automatically assigned the optimal render size
                                        'Fast'   : colors are more washed out
                                        'Medium' : colors are a little washed out (default)
                                        'Slow'   : colors are a little more vivid
                                        'Slower' : colors are more accurate (usually is very slow)
    :param DeepExRefMerge:      Method used by DeepEx to merge the reference frames with the frames propagated by DeepEx.
                                It is applicable only with DeepEx method: 0, 1, 2.
                                Allowed values are:
                                        0 = No RF merge (reference frames can be produced with any frequency) (default)
                                        1 = RF-Merge VeryLow (reference frames are merged with weight=0.3)
                                        2 = RF-Merge Low (reference frames are merged with weight=0.4)
                                        3 = RF-Merge Med (reference frames are merged with weight=0.5)
                                        4 = RF-Merge High (reference frames are merged with weight=0.6)
                                        5 = RF-Merge VeryHigh (reference frames are merged with weight=0.7)
    :param DeepExOnlyRefFrames: If enabled the filter will output in "ScFrameDir" the reference frames. Useful to check
                                and eventually correct the frames with wrong colors (can be used only if DeepExMethod = 0).
                                Default = False
    :param DeepExModel:         Exemplar Model used by DeepEx to propagate color frames.
                                        0 : CMNET2 (default)
                                        1 : Deep-Exemplar
                                        2 : Deep-Remaster
                                        3 : ColorMNet
    :param DeepExVivid:         Depending on selected DeepExModel, if enabled:
                                    0) CMNET2: the saturation will be increased by about 15%.
                                    1) Deep-Exemplar: the saturation will be increased by about 25%.
                                    2) Deep-Remaster: the saturation will be increased by about 20% and Hue by +10.
                                    3) ColorMNet: the frames memory is reset at every reference frame update
                                range [True, False]. Default = False
    :param DeepExEncMode:       Parameter used by CMNET2/ColorMNet to define the encode mode strategy.
                                Available values are:
                                     0: remote encoding. The frames will be colored by a thread outside Vapoursynth.
                                                         This option don't have any GPU memory limitation and will allow
                                                         to fully use the long term frame memory.
                                                         It is the faster encode method (default)
                                     1: local encoding.  The frames will be colored inside the Vapoursynth environment.
                                                         In this case the max_memory will be limited by the size of GPU
                                                         memory (max 15 frames for 24GB GPU).
                                                         Useful for coloring clips with a lot of smooth transitions,
                                                         since in this case is better to use a short frame memory or
                                                         the Deep-Exemplar model, which is faster.
    :param DeepExMaxMemFrames:  Parameter used by CMNET2/ColorMNet/DeepRemaster models.
                                For ColorMNet specify the max number of encoded frames to keep in memory. Default = 0
                                Its value depend on encode mode and must be defined manually following the suggested values.
                                DeepExEncMode=0: there is no memory limit (it could be all the frames in the clip).
                                Suggested values are:
                                    min=50, max=5000
                                If = 0 will be filled with the value of 5000 or the clip length if lower.
                                DeepExEncMode=1: the max memory frames is limited by available GPU memory.
                                Suggested values are:
                                    min=1, max=2      : for 8GB GPU
                                    min=1, max=4      : for 12GB GPU
                                    min=1, max=6      : for 16GB GPU
                                If = 0 will be filled with the max value (depending on total GPU RAM available)
                                For CMNET2/DeepRemaster represent the number to reference frames to keep in memory.
                                Suggested values are:
                                    min=4, max=50 (100 CMNET2)
                                If = 0 will be filled with the value of 20.
    :param ScFrameDir:          if set, define the directory where are stored the reference frames that will be used
                                by "Exemplar-based" Video Colorization models. With DeepExMethod 5,6 this parameter
                                can be the path to a video clip. Default = None
    :param ScThreshold:         Scene change threshold used to generate the reference frames to be used by
                                "Exemplar-based" Video Colorization. It is a percentage of the luma change between
                                the previous and the current frame. range [0-1], default 0.10. If =0 are not generate
                                reference frames.
    :param ScThtOffset:         Offset index used for the Scene change detection. The comparison will be performed,
                                between frame[n] and frame[n-offset]. An offset > 1 is useful to detect blended scene
                                change, range[1, 25]. Default = 1.
    :param ScMinInt:            Minimum number of frame interval between scene changes, range[1, 25]. Default = 1.
    :param ScMinFreq:           if > 0 will be generated at least a reference frame every "ScMinFreq" frames.
                                range [0-1500], default: 0.
    :param ScThtSSIM:           Threshold used by the SSIM (Structural Similarity Index Metric) selection filter.
                                If > 0, will be activated a filter that will improve the scene-change detection,
                                by discarding images that are similar.
                                Suggested values are between 0.35 and 0.65, range [0-1], default 0.0 (deactivated)
    :param ScNormalize:         If true the B&W frames are normalized before use misc.SCDetect(), the normalization will
                                increase the sensitivity to smooth scene changes, range [True, False], default: False
    :param RefRange:            Parameter used only with DeepExMethod in (5, 6). With this parameter it is possible to
                                provide the frame number of clip start and end. For example RefRange=(100, 500)
                                will return the clip's slice: clip[100:500], if RefRange=(0, 0) will be considered all
                                clip's frames.
    :param enable_fp16:         Enable/disable FP16 in ddcolor inference, range [True, False]. Default = True
    :param debug_level:         Set the level of HAVC debug messages. Default = 0 (no messages)
    """
    # disable packages warnings
    disable_warnings()

    HAVC_set_debug_level(debug_level)

    clip, orig_fmt = convert_format_RGB24(clip)

    # Select presets / tuning
    speed_id, deoldify_rf, ddcolor_rf = havc_utils._get_render_factors(Preset)

    # Select VideoTune
    ddcolor_weight = havc_utils._get_mweight(VideoTune)

    # Select Color model
    do_model, dd_model, dd_method = havc_utils._get_color_model(ColorModel)

    if dd_method == 2:
        dd_method = havc_utils._get_comb_method(CombMethod)

    dd_tweak, hue_range, hue_range2, chroma_adjust, chroma_adjust2 = havc_utils._get_color_tune(ColorTune, ColorFix,
                                                                                                ColorMap, dd_model)

    # stabilization is not applicable where are colored only the ref frames or when Denoise is 'None'
    stab_enabled = not DeepExOnlyRefFrames and ColorTune.lower() != 'none'

    # ---------------------- SET ColorTemp & FrameInterp ------------------------------------
    color_temp = havc_utils._get_temp_color(ColorTemp)
    if color_temp > 0:
        ScMinFreq = 1 # Forced to 1
        DeepExVivid = EnableDeepEx  # if EnableDeepEx is true, DeepExVivid is forced to True
    if FrameInterp > 4:
        EnableDeepEx = False  # is not compatible with FrameInterp > 4

    # ---------------------- SET Placebo ------------------------------------
    slices_n = 0
    overlap_x = (round(max(min((0.5 * clip.width)*0.2,192), 64),0) // 2) * 2
    overlap_y = (round(max(min((0.5 * clip.height)*0.2,108), 64),0) // 2) * 2
    deoldify_rf_n = min(max(math.trunc((0.5 * clip.width + overlap_x) / 16), 22), 32)
    ddcolor_rf_n = deoldify_rf_n
    blend_weight = 0 # linear blend good with 20% overlap
    if speed_id in (0, 1): # Placebo, VerySlow
        slices_n = 4 if speed_id == 0 else 2
    # ---------------------- START COLORING --------------------------------------------
    if EnableDeepEx and DeepExMethod in (0, 1, 2, 5, 6, constants.DEF_HAVC_METHOD_PLACEBO):

        havc_utils._check_input(DeepExOnlyRefFrames, ScFrameDir, DeepExMethod, ScThreshold, ScMinFreq, DeepExRefMerge)

        if ScMinFreq > 1:
            ref_freq = ScMinFreq
        else:
            ref_freq = 0

        if DeepExRefMerge > 0:
            ScMinFreq = 1

        if ScThreshold is not None and 0 < ScThreshold < 1:
            ref_tresh = ScThreshold
        else:
            ref_tresh = constants.DEF_THRESHOLD

        if DeepExMethod in (5, 6):
            clip_ref = HAVC_read_video(source=ScFrameDir, fpsnum=clip.fps_num, fpsden=clip.fps_den)

            clip_s: int = RefRange[0]
            clip_e: int = RefRange[1]

            if clip_e > 0 and 0 <= clip_s <= clip_e:
                clip_ref = clip_ref[clip_s: clip_e]

            clip_colored = HAVC_restore_video(clip, clip_ref, render_speed=DeepExPreset, ex_model=DeepExModel,
                                              ref_merge=DeepExRefMerge, ref_thresh=ref_tresh, ref_freq=ref_freq,
                                              max_memory_frames=DeepExMaxMemFrames, render_vivid=DeepExVivid,
                                              encode_mode=DeepExEncMode, ref_norm=ScNormalize)

        else: # HAVC method in (0, 1, 2, DEF_HAVC_METHOD_PLACEBO)

            if FrameInterp == 0 or DeepExRefMerge == 0:
                clip_ref = HAVC_colorizer(clip, method=dd_method, mweight=ddcolor_weight,
                                      deoldify_p=[do_model, deoldify_rf, 1.0, 0.0],
                                      ddcolor_p=[dd_model, ddcolor_rf, 1.0, 0.0, enable_fp16],
                                      ddtweak=dd_tweak, ddtweak_p=[constants.DEF_TWEAK_p, hue_range],
                                      sc_threshold=ScThreshold, sc_tht_offset=ScThtOffset, sc_min_freq=ScMinFreq,
                                      sc_min_int=ScMinInt, sc_tht_ssim=ScThtSSIM, sc_normalize=ScNormalize,
                                      debug_level=debug_level)
            else:
                clip_ref = HAVC_colorizer_fast(clip, method=dd_method, mweight=ddcolor_weight,
                                    deoldify_p=[do_model, deoldify_rf, 1.0, 0.0],
                                    ddcolor_p=[dd_model, ddcolor_rf, 1.0, 0.0, enable_fp16],
                                    ddtweak=dd_tweak, ddtweak_p=[constants.DEF_TWEAK_p, hue_range],
                                    frame_interp=FrameInterp, chroma_adjust=chroma_adjust, debug_level=debug_level)
            if color_temp > 0:
                clip_ref = HAVC_cmnet1(clip=clip, clip_ref=clip_ref, render_speed='Medium', render_vivid=True,
                                       ref_merge=color_temp, dark=True, dark_p=[0.2, 0.8], ref_thresh=0.10,
                                       encode_mode=0, max_memory_frames=1, ref_freq=0, ref_norm=True, smooth=True,
                                       smooth_p=[0.3, 0.7, 0.9, 0.0, "none"], colormap=chroma_adjust)

            if DeepExMethod != constants.DEF_HAVC_METHOD_PLACEBO:
                clip_colored = HAVC_deepex(clip=clip, clip_ref=clip_ref, method=DeepExMethod, render_speed=DeepExPreset,
                                       render_vivid=DeepExVivid, ref_merge=DeepExRefMerge, sc_framedir=ScFrameDir,
                                       only_ref_frames=DeepExOnlyRefFrames, dark=True, dark_p=[0.2, 0.8],
                                       ref_thresh=ref_tresh, ex_model=DeepExModel, encode_mode=DeepExEncMode,
                                       max_memory_frames=DeepExMaxMemFrames, ref_freq=ScMinFreq, ref_norm=ScNormalize,
                                       smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"], colormap=chroma_adjust)
            else:
                clip_colored = clip_ref

        # are applied the faster stabilization settings
        if DeepExMethod != constants.DEF_HAVC_METHOD_PLACEBO:
            clip_colored = HAVC_stabilizer(clip_colored, stab=stab_enabled, stab_p=[3, 'A', 1, 0, 0, 0],
                                       colormap=chroma_adjust2)

    elif EnableDeepEx and DeepExMethod in (3, 4):

        if DeepExModel == 0:
            clip_colored = HAVC_cmnet2(clip, clip_ref=None, method=DeepExMethod, render_speed=DeepExPreset,
                render_vivid=DeepExVivid, ref_merge=0, sc_framedir=ScFrameDir, ref_norm=False, dark=False,
                smooth=False, colormap=chroma_adjust, ref_weight=None, ref_thresh=None, ref_freq=None,
                encode_mode=DeepExEncMode, max_memory_frames=DeepExMaxMemFrames)
        elif DeepExModel == 2:
            # call to faster version of DeepRemaster that read directly the images folder (ref_mode=0)
            clip_colored = HAVC_DeepRemaster(clip, render_vivid=DeepExVivid, ref_dir=ScFrameDir,
                                             ref_buffer_size=DeepExMaxMemFrames, ref_mode=0)

        else:
            ref_merge = 0 if DeepExModel != 3 else DeepExRefMerge
            clip_colored = HAVC_deepex(clip=clip, clip_ref=None, method=DeepExMethod, render_speed=DeepExPreset,
                                       render_vivid=DeepExVivid, sc_framedir=ScFrameDir, ref_merge=ref_merge,
                                       only_ref_frames=DeepExOnlyRefFrames, dark=True, dark_p=[0.2, 0.8],
                                       smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"], ex_model=DeepExModel,
                                       encode_mode=DeepExEncMode, max_memory_frames=DeepExMaxMemFrames,
                                       colormap=chroma_adjust)

    else:  # No DeepEx -> HAVC classic

        if FrameInterp == 0:
            if slices_n == 0:
                clip_colored = HAVC_colorizer(clip, method=dd_method, mweight=ddcolor_weight,
                                      deoldify_p=[do_model, deoldify_rf, 1.0, 0.0],
                                      ddcolor_p=[dd_model, ddcolor_rf, 1.0, 0.0, enable_fp16],
                                      ddtweak=dd_tweak, ddtweak_p=[constants.DEF_TWEAK_p, hue_range])
            else:
                clips = HAVC_clip_slice(clip, slices=slices_n, overlap_x=overlap_x, overlap_y=overlap_y)
                for i in range(slices_n):
                    #img_bw=frame_to_image(clips.tiles[i].get_frame(0))
                    clips.tiles[i] = HAVC_colorizer(clips.tiles[i], method=dd_method, mweight=ddcolor_weight,
                                      deoldify_p=[do_model, deoldify_rf_n, 1.0, 0.0],
                                      ddcolor_p=[dd_model, ddcolor_rf_n, 1.0, 0.0, enable_fp16],
                                      ddtweak=dd_tweak, ddtweak_p=[constants.DEF_TWEAK_p, hue_range])
                    #img_col = frame_to_image(clips.tiles[i].get_frame(0))
                clip_colored = HAVC_clip_reconstruct(clips, blend_weight=blend_weight, chroma_resize=True)
                #img_col2 = frame_to_image(clip_colored.get_frame(0))
        else:
            if slices_n == 0:
                clip_colored = HAVC_colorizer_fast(clip, method=dd_method, mweight=ddcolor_weight,
                                      deoldify_p=[do_model, deoldify_rf, 1.0, 0.0],
                                      ddcolor_p=[dd_model, ddcolor_rf, 1.0, 0.0, enable_fp16],
                                      ddtweak=dd_tweak, ddtweak_p=[constants.DEF_TWEAK_p, hue_range],
                                      frame_interp=FrameInterp, chroma_adjust=chroma_adjust, debug_level=debug_level)
            else:
                clips = HAVC_clip_slice(clip, slices=slices_n, overlap_x=overlap_x, overlap_y=overlap_y)
                for i in range(slices_n):
                    clips.tiles[i] = HAVC_colorizer_fast(clip, method=dd_method, mweight=ddcolor_weight,
                                      deoldify_p=[do_model, deoldify_rf_n, 1.0, 0.0],
                                      ddcolor_p=[dd_model, ddcolor_rf_n, 1.0, 0.0, enable_fp16],
                                      ddtweak=dd_tweak, ddtweak_p=[constants.DEF_TWEAK_p, hue_range],
                                      frame_interp=FrameInterp, chroma_adjust=chroma_adjust, debug_level=debug_level)
                clip_colored = HAVC_clip_reconstruct(clips, blend_weight=blend_weight, chroma_resize=True)
        if color_temp > 0:
            clip_colored = clip_colored.std.SetFrameProp(prop="sc_threshold", floatval=0.1)
            clip_colored = clip_colored.std.SetFrameProp(prop="sc_frequency", intval=1)
            clip_colored = HAVC_cmnet2(clip=clip, clip_ref=clip_colored, render_speed='Medium', render_vivid=False,
                                   ref_merge=color_temp, dark=True, dark_p=[0.2, 0.8], ref_thresh=0.10,
                                   encode_mode=0, max_memory_frames=1, ref_freq=0, ref_norm=True, smooth=True,
                                   smooth_p=[0.3, 0.7, 0.9, 0.0, "none"], colormap=chroma_adjust)

        if speed_id > 4:  # 'fast', 'faster', 'veryfast' -> is used only colormap
            clip_colored = HAVC_stabilizer(clip_colored, colormap=chroma_adjust)
        elif speed_id > 1:  # 'slower', 'slow', 'medium' -> are used all the filters except hue_range2 and stab (deoldify only)
            if dd_method == 0:
                clip_colored = HAVC_stabilizer(clip_colored, dark=True, dark_p=[0.2, 0.8], colormap=chroma_adjust,
                                               smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"],
                                               stab=False)
            else:
                clip_colored = HAVC_stabilizer(clip_colored, dark=True, dark_p=[0.2, 0.8], colormap=chroma_adjust,
                                               smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"],
                                               stab=stab_enabled, stab_p=[5, 'A', 1, 15, 0.2, 0.8])
        else:  # 'placebo', 'veryslow'  -> are used all the filters
            clip_colored = HAVC_stabilizer(clip_colored, dark=True, dark_p=[0.2, 0.8], colormap=chroma_adjust,
                                           smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"],
                                           stab=stab_enabled, stab_p=[5, 'A', 1, 15, 0.2, 0.8, hue_range2])

    return restore_format(clip_colored, orig_fmt)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Color post process  filter for restoring the correct colors for clip previously 
colored using: DDColor & Zhang's models. 
"""

def HAVC_rgb_denoise(clip: vs.VideoNode, denoise_levels: list[float] = (0.4, 0.3),
                     rgb_factors: list[float] = (0.95, 1.05, 1.01)) -> vs.VideoNode:
    """HAVC Color Post Processing function for restoring the correct colors for clip previously colored
       using: DDColor & Zhang's models.

       :param clip:                clip to process, any format is supported.
       :param denoise_levels:      denoise level for colors and contrast
                                       [0] color denoise, range[0,1] (default = 0.4)
                                       [1] contrast denoise, range[0,1] (default = 0.3)
       :param rgb_factors:         rgb adjustment factors
                                       [0] RED adjustment factor, range[0,1] (default = 0.95)
                                       [1] GREEN adjustment factor, range[0,1] (default = 1.05)
                                       [2] BLUE adjustment factor, range[0,1] (default = 1.01)

    """

    clip, orig_fmt = convert_format_RGB24(clip)

    clip = rgb_denoise(clip, denoise_levels, rgb_factors)

    return restore_format(clip, orig_fmt)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Color post process  filter for restoring the color of clip previously colored 
with HAVC by improving contrast and luminosity of colored frames. 
It is a wrapper to HAVC_main_restore() 
"""

def HAVC_ColorAdjust(clip: vs.VideoNode, BlackWhiteTune: str = 'Light', BlackWhiteMode: int = 0,
                     BlackWhiteBlend: bool = True, ReColor: bool = True, Strength: int = 0, ScThreshold: float = 0.10,
                     ScNormalize: bool = True, DeepExVivid: bool = False,  ScMinFreq: int = 0,
                     chroma_resize: bool = False) -> vs.VideoNode:
    """HAVC Color Post Processing function

        :param clip:               clip to process any clip format is supported.
        :param BlackWhiteTune:     This parameter allows to improve contrast and luminosity of frames colored with HAVC.
                                   Allowed values are:
                                        'None',
                                        'Light', (default)
                                        'Medium', 
                                        'Strong'
        :param BlackWhiteMode:     Method used by BlackWhiteTune to perform colors adjustments.
                                   Allowed values are:
                                        0 : CLAHE (luma) (default)
                                        1 : Simple (RGB)
                                        2 : CLAHE (RGB)
                                        3 : CLAHE (luma) + Simple (RGB)
                                        4 : ScaleAbs – LUT
                                        5 : Multi-Scale Retinex (HAVC)
                                        6 : Multi-Scale Retinex (B&W)
        :param BlackWhiteBlend:    If enabled the frames adjusted with BlackWhiteTune will be blended with the original frames.
                                   Default = True 
        :param ReColor:            If True the clip will be re-colored with ColorMNet to enforce color temporal
                                   stabilization. To be used if the clip was colored using an AI automatic video
                                   colorizer like HAVC. Default = True
        :param Strength:           Color temporal stabilization strength, using high level the colors will be more
                                   stable but will be also more washed.
                                   Allowed values are:
                                        0 = VeryLow (default)
                                        1 = Low
                                        2 = Med
                                        3 = High
                                        4 = VeryHigh
        :param ScThreshold:        Scene change threshold used to generate the reference frames to be used by
                                   ColorMNet. It is a percentage of the luma change between the previous and the
                                   current frame. range [0-1], default 0.10. If =0 are not generate reference frames.
                                   default = 0.10
        :param ScNormalize:        If true the frames are normalized before using misc.SCDetect(), the normalization
                                   will increase the sensitivity to smooth scene changes, range [True, False],
                                   default: True
        :param DeepExVivid:        if enabled the saturation will be increased by about 15% (CMNET2).
                                   range [True, False], default: False
        :param ScMinFreq:          if > 0 will be generated at least a reference frame every "ScMinFreq" frames.
                                   range [0-1500], default: 0.
        :param chroma_resize:      If True, the clip will be downscaled before applying the filter to speed up
                                   the processing, default = False
    """
    # disable packages warnings
    disable_warnings()

    DeepExModel: int = 3   # ColorMNet
    DeepExRefMerge: int = 1 + min(max(4 - Strength, 0), 4)
    DeepExPreset: str = 'medium'
    DeepExMaxMemFrames: int = 0
    DeepExMethod: int = 5
    DeepExEncMode: int = 0

    if BlackWhiteTune.lower() == 'none' and not ReColor:
        return clip

    if not isinstance(clip, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "ColorPostProcessing: this is not a clip")

    clip, orig_fmt = convert_format_RGB24(clip, chroma_resize=chroma_resize)

    if ReColor:
        clip_colored = clip
        clip_colored = clip_colored.std.SetFrameProp(prop="sc_threshold", floatval=0.1)
        clip_colored = clip_colored.std.SetFrameProp(prop="sc_frequency", intval=1)
    else:
        clip_colored = None

    tn_id = _get_tune_id(BlackWhiteTune)
    if tn_id != 0 and BlackWhiteMode in (4, 6):
        # redefine color mapping
        bw_tune = 'none'
        bw_mode = 4
    else:
        bw_tune = BlackWhiteTune
        bw_mode = BlackWhiteMode

    clip_restored = HAVC_main_restore(clip, clip_colored, DeepExPreset, DeepExModel, DeepExRefMerge,
                                      ScThreshold, ScMinFreq, ScNormalize, DeepExMaxMemFrames, DeepExMethod,
                                      DeepExVivid, DeepExEncMode, BlackWhiteTune=bw_tune,
                                      BlackWhiteMode=bw_mode, BlackWhiteBlend=BlackWhiteBlend)

    if tn_id != 0 and BlackWhiteMode in (4, 6):
        # apply new color mapping
        if BlackWhiteMode == 4 and tn_id == 1:
            clip_restored = vs_timecube(clip_restored, strength=0.8, lut_effect=constants.DEF_LUT_Exploration)
        elif BlackWhiteMode == 4 and tn_id == 2:
            clip_restored = vs_timecube(clip_restored, strength=0.6, lut_effect=constants.DEF_LUT_City_Skyline)
        elif BlackWhiteMode == 4 and tn_id == 3:
            clip_restored = vs_timecube(clip_restored, strength=0.5, lut_effect=constants.DEF_LUT_Amber_Light)
        if BlackWhiteMode == 6 and tn_id == 1:
            clip_restored = vs_timecube(clip_restored, strength=0.6, lut_effect=constants.DEF_LUT_FUJ_Film)
        elif BlackWhiteMode == 6 and tn_id == 2:
            clip_restored = vs_timecube(clip_restored, strength=0.7, lut_effect=constants.DEF_LUT_Flat_Pop)
        elif BlackWhiteMode == 6 and tn_id == 3:
            clip_restored = vs_timecube(clip_restored, strength=0.5, lut_effect=constants.DEF_LUT_Warm_Haze)

    if ReColor:
        clip_restored = vs_reduce_flicker(clip_restored)

    return restore_format(clip_restored, orig_fmt)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Wrapper to the patched filter Retinex MSRCP 
"""
def HAVC_retinex(clip: vs.VideoNode, luma_dark: float = constants.DEF_RETINEX_DARK, luma_bright: float = constants.DEF_RETINEX_BRIGHT,
                 sigmas: list[float]=(25, 80, 250), range_tv_in: bool = True, range_tv_out: bool=True,
                 blend: bool = False, chroma_resize: bool = False) -> vs.VideoNode:
    """patched filter Retinex MSRCP to avoid artifacts on dark/bright frames

           :param clip:           clip to process, any clip format is supported.
           :param luma_dark:      luma level to identify dark frames, range [0-1], default = 0.20
           :param luma_bright:    luma level to identify bright frames, range [0-1], default = 0.80
           :param sigmas:         sigma of Gaussian function to apply Gaussian filtering.
                                  Assign an array of multiple sigma to apply MSR. Default = [25, 80, 250]
                                  Basically, in SSR(Single Scale Retinex), small sigma result in stronger dynamic
                                  range compression and local contrast enhancement, while large sigma result in
                                  better color rendition. To afford an acceptable trade-off between these features,
                                  MSR combines different scales to compute the final Retinex output.
           :param range_tv_in:    Determine the value range of input clip. True means full range/PC range,
                                  and False means limited range/TV range. Default = True
           :param range_tv_out:   Determine the value range of output clip. True means full range/PC range,
                                  and False means limited range/TV range. Default = True
           :param blend:          If True the dark frames of filtered clip will be blended with the input clip.
                                  Default = False
           :param chroma_resize:  If True, the clip will be downscaled before applying the filter to speed up
                                   the processing, default = False
    """

    clip, orig_fmt = convert_format_RGB24(clip, chroma_resize=chroma_resize)

    clip = vs_retinex(clip, luma_dark, luma_bright, sigmas, range_tv_in, range_tv_out, blend, fast_mode=True)

    return restore_format(clip, orig_fmt)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Post process  filter for restoring the color of clip previously colored with HAVC
by improving contrast and luminosity of colored frames
with HAVC.
"""

def HAVC_main_restore(clip: vs.VideoNode, clip_colored: vs.VideoNode | None, DeepExPreset: str = 'medium',
                      DeepExModel: int = 0, DeepExRefMerge: int = 0, ScThreshold: float = constants.DEF_THRESHOLD,
                      ScMinFreq: int = 0, ScNormalize: bool = False, DeepExMaxMemFrames: int = 0, DeepExMethod: int = 5,
                      DeepExVivid: bool = False, DeepExEncMode: int = 0, BlackWhiteTune: str = 'Medium',
                      BlackWhiteMode: int = 0, BlackWhiteBlend: bool = True, chroma_resize: bool = False) -> vs.VideoNode:
    """Main HAVC restoring function

        :param clip:               clip to process, any clip format is supported.
        :param clip_colored:       Clip containing the colored frames to be restored
        :param BlackWhiteTune:     This parameter allows to improve contrast and luminosity of frames colored with HAVC.
                                   Allowed values are:
                                        'None',
                                        'Light',
                                        'Medium', (default)
                                        'Strong'
        :param BlackWhiteMode:     Method used by BlackWhiteTune to perform colors adjustments.
                                   Allowed values are:
                                        0 : CLAHE (luma) (default)
                                        1 : Simple (RGB)
                                        2 : CLAHE (RGB)
                                        3 : CLAHE (luma) + Simple (RGB)
                                        4 : ScaleAbs – LUT
                                        5 : Multi-Scale Retinex (HAVC)
                                        6 : Multi-Scale Retinex (B&W)
        :param BlackWhiteBlend:    If enabled the frames adjusted with BlackWhiteTune will be blended with the original frames.
        :param DeepExMethod:       Method to use to generate reference frames.
                                            0 = HAVC same as video (default)
                                            1 = HAVC + RF same as video
                                            2 = HAVC + RF different from video
                                            3 = external RF same as video
                                            4 = external RF different from video
                                            5 = external ClipRef same as video
                                            6 = external ClipRef different from video
        :param DeepExPreset:       Preset to control the render method and speed:
                                   Allowed values are:
                                            'Fast'   (colors are more washed out)
                                            'Medium' (colors are a little washed out)
                                            'Slow'   (colors are a little more vivid)
        :param DeepExRefMerge:     Method used by DeepEx to merge the reference frames with the frames propagated by DeepEx.
                                   It is applicable only with DeepEx method: 0, 1, 2.
                                   Allowed values are:
                                            0 = No RF merge (reference frames can be produced with any frequency) (default)
                                            1 = RF-Merge VeryLow (reference frames are merged with weight=0.3)
                                            2 = RF-Merge Low (reference frames are merged with weight=0.4)
                                            3 = RF-Merge Med (reference frames are merged with weight=0.5)
                                            4 = RF-Merge High (reference frames are merged with weight=0.6)
                                            5 = RF-Merge VeryHigh (reference frames are merged with weight=0.7)
        :param DeepExModel:        Exemplar Model used by DeepEx to propagate color frames.
                                            0 : CMNET2 (default)
                                            1 : Deep-Exemplar
                                            2 : Deep-Remaster
                                            3 : ColorMNet
        :param DeepExVivid:        Depending on selected DeepExModel, if enabled:
                                        0) CMNET2: the saturation will be increased by about 15%.
                                        1) Deep-Exemplar: the saturation will be increased by about 25%.
                                        2) Deep-Remaster: the saturation will be increased by about 20% and Hue by +10.
                                        3) ColorMNet: the frames memory is reset at every reference frame update
                                    range [True, False]. Default: False
        :param DeepExEncMode:      Parameter used by CMNET2/ColorMNet to define the encode mode strategy.
                                   Available values are:
                                         0: remote encoding. The frames will be colored by a thread outside Vapoursynth.
                                                             This option don't have any GPU memory limitation and will allow
                                                             to fully use the long term frame memory.
                                                             It is the faster encode method (default)
                                         1: local encoding.  The frames will be colored inside the Vapoursynth environment.
                                                             In this case the max_memory will be limited by the size of GPU
                                                             memory (max 15 frames for 24GB GPU).
                                                             Useful for coloring clips with a lot of smooth transitions,
                                                             since in this case is better to use a short frame memory or
                                                             the Deep-Exemplar model, which is faster.
                                         2: remote all-ref   Same as "remote encoding" but all the available reference frames
                                                             will be used for the inference at the beginning of encoding.
        :param DeepExMaxMemFrames: Parameter used by ColorMNet/DeepRemaster models.
                                   For ColorMNet specify the max number of encoded frames to keep in memory.
                                   Its value depend on encode mode and must be defined manually following the suggested values.
                                   DeepExEncMode=0: there is no memory limit (it could be all the frames in the clip).
                                   Suggested values are:
                                        min=150, max=10000
                                   If = 0 will be filled with the value of 10000 or the clip length if lower.
                                   DeepExEncMode=1: the max memory frames is limited by available GPU memory.
                                   Suggested values are:
                                        min=1, max=4      : for 8GB GPU
                                        min=1, max=8      : for 12GB GPU
                                        min=1, max=15     : for 24GB GPU
                                   If = 0 will be filled with the max value (depending on total GPU RAM available)
                                   For DeepRemaster represent the number to reference frames to keep in memory.
                                   Suggested values are:
                                        min=4, max=50
                                   If = 0 will be filled with the value of 20.
        :param ScThreshold:        Scene change threshold used to generate the reference frames to be used by
                                   "Exemplar-based" Video Colorization. It is a percentage of the luma change between
                                   the previous and the current frame. range [0-1], default 0.10. If =0 are not generate
                                   reference frames.
        :param ScMinFreq:          if > 0 will be generated at least a reference frame every "ScMinFreq" frames.
                                   range [0-1500], default: 0.
        :param ScNormalize:        If true the B&W frames are normalized before use misc.SCDetect(), the normalization will
                                   increase the sensitivity to smooth scene changes, range [True, False], default: False
        :param chroma_resize:      If True, the clip will be downscaled before applying the filter to speed up the processing.
    """

    clip, orig_fmt = convert_format_RGB24(clip, chroma_resize=chroma_resize)

    BWTuneRetinex: bool = BlackWhiteTune.lower() != "none" and BlackWhiteMode == 6

    if not clip_colored == None:
        if BWTuneRetinex:
            clip = HAVC_bw_tune(clip, bw_tune=BlackWhiteTune, bw_method=5, luma_blend=BlackWhiteBlend)
            BlackWhiteTune = "none"
            BlackWhiteMode = 5
        clip = HAVC_restore_video(clip, clip_colored, method=DeepExMethod, render_speed=DeepExPreset, ex_model=DeepExModel,
                       ref_merge=DeepExRefMerge, ref_thresh=ScThreshold, ref_freq=ScMinFreq,
                       max_memory_frames=DeepExMaxMemFrames, render_vivid=DeepExVivid,
                       encode_mode=DeepExEncMode, ref_norm=ScNormalize)
        if BWTuneRetinex:
            clip = HAVC_tweak(clip, hue=5.0, sat=0.95, bright=0, cont=0.98, gamma=0.98)
        elif BlackWhiteTune.lower() != "none":
            clip = HAVC_adjust_rgb(clip, strength=0.5, gamma=[1.0, 1.0, 0.98])
            clip = HAVC_tweak(clip, hue=5, sat=1.05, bright=0, cont=1.0)
            return restore_format(clip, orig_fmt)
    # ------------------------------------------------------------------------
    if BlackWhiteTune.lower() == 'none':
        return restore_format(clip, orig_fmt)
    else:
        BlackWhiteMode = min(BlackWhiteMode, 5)

    i = BlackWhiteMode
    cont = [1.0, 0.95, 1.0, 0.95, 0.95, 0.90]
    hue = [-10.0, -10.0, -10.0, -10.0, -10.0, -5.0]
    sat = [1.10, 1.05, 1.10, 1.10, 0.95, 0.95]
    bright = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

    if BlackWhiteTune.lower() == 'light':
        gamma = [1.0, 0.98, 0.98, 0.98, 0.98, 0.98]
    else:
        gamma = [1.0, 0.95, 0.95, 0.95, 0.95, 0.95]

    clip = HAVC_bw_tune(clip, BlackWhiteTune, i, BlackWhiteBlend, True)

    if BlackWhiteMode < 4:  # Skip Retinex, ScaleAbs
        clip = HAVC_tweak(clip, hue[i], sat[i], bright[i], cont[i], gamma[i])

    return restore_format(clip, orig_fmt)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Post process  filter for improving colors and contrast/luminosity of clips colored
with HAVC.
"""


def HAVC_bw_tune(clip: vs.VideoNode = None, bw_tune: str = 'Light', bw_method: int = 0,
                 luma_blend: bool = True, range_tv: bool = True, chroma_resize: bool = False) -> vs.VideoNode:
    """Post process filter for improving colors and contrast/luminosity of colored clips with HAVC

    :param clip:          Clip to process, any clip format is supported.
    :param bw_tune:       This parameter allows to improve contrast and luminosity of input clip colored with HAVC.
                          Allowed values are:
                              'None'
                              'Light', (default)
                              'Medium', 
                              'Strong'
    :param bw_method:     Method used to perform color adjustments.
                          Allowed values are:
                                0 : CLAHE (luma) (default)
                                1 : Simple (RGB)
                                2 : CLAHE (RGB)
                                3 : CLAHE (luma) + Simple (RGB)
                                4 : ScaleAbs – LUT
                                5 : Multi-Scale Retinex (HAVC)
    :param luma_blend:    If enabled the equalized image is blended with the original image, darker is the image and more
                          weight will be assigned to the original image. default = True
    :param range_tv:      If True, perform the color adjustments on limited TV range (the filter works better in TV range).
    :param chroma_resize: If True, the clip will be downscaled before applying the filter to speed up the processing.
    """

    clip, orig_fmt = convert_format_RGB24(clip, chroma_resize=chroma_resize)

    bw_tune_p = bw_tune.lower()
    bw_tune = ['none', 'light', 'medium', 'strong']
    b_strength = [0.0, 0.30, 0.40, 0.50]
    w_strength = [0.0, 0.30, 0.40, 0.50]
    r_factor = [1.0, 0.96, 0.94, 0.92]
    g_factor = [1.0, 1.03, 1.05, 1.08]
    b_factor = [1.0, 1.0, 1.0, 1.0]

    bw_method = min(5, bw_method)

    if bw_method == 5:
        b_strength = [0.0, 0.98, 0.99, 1.0]

    bw_id = 0
    try:
        bw_id = bw_tune.index(bw_tune_p)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_bw_tune: B&W tune choice is invalid: ", bw_tune_p)

    if bw_id == 0:
        return clip

    if bw_method == 4:
        weight3 = float(bw_id)
    else:
        weight3 = w_strength[bw_id]

    r =  r_factor[bw_id]
    g = g_factor[bw_id]
    b = b_factor[bw_id]

    if range_tv:
        clip = clip.std.Levels(min_in=0, max_in=255, min_out=16, max_out=235)
        clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_in_s="full", range_s="limited")

    # step #1 : rgb colors are normalized and changed using rgb factors (this will change also the contrast/luminosity)
    if bw_method < 4: # Skip Retinex, ScaleAbs
        clip = rgb_balance(clip=clip, strength=w_strength[bw_id], rgb_factor=[r, g, b])
    # step #2 : the contrast/luminosity previously changed are adjusted/fixed using histogram equalization
    clip = rgb_equalizer(clip=clip, method=bw_method, strength=b_strength[bw_id], weight3=weight3,
                         luma_blend=luma_blend, range_tv=range_tv)

    if range_tv:
        clip = clip.std.Levels(min_in=16, max_in=235, min_out=0, max_out=255)
        clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_in_s="limited", range_s="full")

    return restore_format(clip, orig_fmt)


def HAVC_adjust_rgb(clip: vs.VideoNode = None, strength: float = 0.0, factor: list = (1.0, 1.0, 1.0),
                    bias: list = (0, 0, 0), gamma: list = (1.0, 1.0, 1.0)) -> vs.VideoNode:
    """Utility function to change the color and luminance of RGB clip.
       Gain, bias (offset) and gamma can be set independently on each channel.

       :param clip:         Clip to process, any format is supported.
                            RGB adjustments.
       :param strength:     This parameter allows to control the strength of the RGB normalization, strength=0 is
                            equivalent to no normalization, while with strength=1 will be applied 100% normalization.
                            The RGB normalization is applied before the other RGB adjustments. Range [0, 1], default=0
       :param factor:       List of Red, green and blue scaling factor, in the list format: (r, g, b).
                            Range 0.0 to 255.0, default = (1, 1, 1).
                            For example, r=1.3 multiplies the red channel pixel values by 1.3.
       :param bias:         List of Red, green and blue bias adjustments, in the list format: (rb, gb, bb).
                            Bias adjustment—add a fixed positive or negative value to a channel's pixel values.
                            For example, rb=16 will add 16 to all red pixel values and rb=-32 will subtract 32 from all
                            red pixel values, default = (0, 0, 0).
       :param gamma:        List of Red, green and blue gamma adjustments, in the list format: (rg, gg, bg).
                            Gamma adjustment—an exponential gain factor. For example, rg=1.2 will brighten the red
                            pixel values and gg=0.8 will darken the green pixel values.
    """

    clip, orig_fmt = convert_format_RGB24(clip)

    if strength == 1:
        clip = vs_rgb_normalize(clip)
    elif 0 < strength < 1:
        rgb = vs_rgb_normalize(clip)
        clip = vs_simple_merge(clip, rgb, weight=strength)

    clip_new = havc_utils.adjust_rgb(clip, factor, bias, gamma)

    return restore_format(clip_new, orig_fmt)


def HAVC_tweak(clip: vs.VideoNode = None, hue: float = 0, sat: float = 1, bright: float = 0,
               cont: float = 1, gamma: float = 1) -> vs.VideoNode:
    """Pre/post - process filter for adjust: hue, saturation, brightness, contrast and gamma of a video clip

        :param clip:        Clip to process, any format is supported.
        :param hue:         Adjust the color hue of the image.
                                 hue>0.0 shifts the image towards red.
                                 hue<0.0 shifts the image towards green.
                            Range -180.0 to +180.0, default = 0.0
        :param sat:         Adjust the color saturation of the image by controlling gain of the color channels.
                                 sat>1.0 increases the saturation.
                                 sat<1.0 reduces the saturation.
                            Use sat=0 to convert to GreyScale.
                            Range 0.0 to 10.0, default = 1.0
        :param bright:      Change the brightness of the image by applying a constant bias to the luma channel.
                                 bright>0.0 increases the brightness.
                                 bright<0.0 decreases the brightness.
                            Range -255.0 to 255.0, default = 0.0
        :param cont:        Change the contrast of the image by multiplying the luma values by a constant.
                                 cont>1.0 increase the contrast (the luma range will be stretched).
                                 cont<1.0 decrease the contrast (the luma range will be contracted).
                            Range 0.0 to 10.0, default = 1.0
        :param gamma:       Change the gamma of image which controls the degree of non-linearity in the luma
                            correction. Higher gamma brightens the output; lower gamma darkens the output.
                            Range -10.0 to 10.0, default = 1.0
    """

    clip, orig_fmt = convert_format_RGB24(clip)

    clip_new = vs_tweak(clip, hue=hue, sat=sat, bright=bright, cont=cont, gamma=gamma)

    return restore_format(clip_new, orig_fmt)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Exemplar-based coloring function with additional post-process filters 
"""


def HAVC_deepex(clip: vs.VideoNode = None, clip_ref: vs.VideoNode = None, method: int = 0, render_speed: str = 'medium',
                render_vivid: bool = False, ref_merge: int = 0, sc_framedir: str = None, ref_norm: bool = False,
                only_ref_frames: bool = False, dark: bool = False, dark_p: list = (0.2, 0.8), smooth: bool = False,
                smooth_p: list = (0.3, 0.7, 0.9, 0.0, "none"), colormap: str = "none", ref_weight: float = None,
                ref_thresh: float = None, ref_freq: int = None, ex_model: int = 0, encode_mode: int = 0,
                max_memory_frames: int = 0, retry_threshold: float = 0, retry_model: int = 0,
                high_resolution: bool = False, torch_dir: str = model_dir) -> vs.VideoNode:
    """Towards Video-Realistic Colorization via Exemplar-based framework

    :param clip:                Clip to process, any format is supported
    :param clip_ref:            Clip containing the reference frames (necessary if method=0,1,2,5,6)
    :param method:              Method to use to generate reference frames (RF).
                                        0 = HAVC same as video (default)
                                        1 = HAVC + RF same as video
                                        2 = HAVC + RF different from video
                                        3 = external RF same as video
                                        4 = external RF different from video
                                        5 = external ClipRef same as video
                                        6 = external ClipRef different from video
    :param render_speed:        Preset to control the render method and speed:
                                Allowed values are:
                                        'Auto'   : will be automatically assigned the optimal render size (default)
                                        'Fast'   : colors are more washed out
                                        'Medium' : colors are a little washed out
                                        'Slow'   : colors are a little more vivid
                                        'Slower' : colors are more accurate (usually is very slow)
    :param render_vivid:        Depending on selected ex_model, if enabled (True):
                                    0) CMNET2: the saturation will be increased by about 15%.
                                    1) Deep-Exemplar: the saturation will be increased by about 25%.
                                    2) Deep-Remaster: the saturation will be increased by about 20% and Hue by +10.
                                    3) ColorMNet: the frames memory is reset at every reference frame update
                                range [True, False]. Default = False
    :param ref_merge:           Method used by DeepEx to merge the reference frames with the frames propagated by DeepEx.
                                It is applicable only with DeepEx method: 0, 1, 5.
                                The HAVC reference frames must be produced with frequency = 1.
                                Allowed values are:
                                        0 = No RF merge (reference frames can be produced with any frequency)
                                        1 = RF-Merge VeryLow (reference frames are merged with weight=0.3)
                                        2 = RF-Merge Low (reference frames are merged with weight=0.4)
                                        3 = RF-Merge Med (reference frames are merged with weight=0.5)
                                        4 = RF-Merge High (reference frames are merged with weight=0.6)
                                        5 = RF-Merge VeryHigh (reference frames are merged with weight=0.7)
                                >> ref_merge is used also by ex_model=3, in this case it represent the weight that will
                                be used to merge the models.
    :param ref_weight:          If (ref_merge > 0), represent the weight used to merge the reference frames.
                                If is not set, is assigned automatically a value depending on ref_merge/method values.
    :param ref_thresh:          Represent the threshold used to create the reference frames. If is not set, is assigned
                                automatically a value of 0.10
    :param ref_freq:            If > 0 will be generated at least a reference frame every "ref_freq" frames.
                                range [0-1500]. If is not set, is assigned automatically a value depending on
                                ref_merge/method values.
    :param ref_norm:            If true the B&W frames are normalized before apply the Scene Detection to generate the
                                reference frames. The normalization will increase the sensitivity to smooth scene changes,
                                range [True, False], default: False
    :param sc_framedir:         If set, define the directory where are stored the reference frames. If only_ref_frames=True,
                                and method=0 this directory will be written with the reference frames used by the filter.
                                if method!=0 the directory will be read to create the reference frames that will be used
                                by "Exemplar-based" Video Colorization. The reference frame name must be in the
                                format: ref_nnnnnn.[jpg|png], for example the reference frame 897 must be
                                named: ref_000897.png. With methods 5,6 this parameters can be the path to a video clip.
    :param only_ref_frames:     If enabled the filter will output in "sc_framedir" the reference frames. Useful to check
                                and eventually correct the frames with wrong colors.
    :param dark:                Enable/disable darkness filter (only on ref-frames), range [True,False]
    :param dark_p:              Parameters for darken the clip's dark portions, which sometimes are wrongly colored by the color models
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0.1-0.5] (0.01=1%)
                                      [1] : dark_amount: amount of desaturation to apply to the dark area, range [0-1]
                                      [2] : "chroma range" parameter (optional), if="none" is disabled (see the README)
    :param smooth:              Enable/disable chroma smoothing (only on ref-frames), range [True, False]
    :param smooth_p:            parameters to adjust the saturation and "vibrancy" of the clip.
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0-1] (0.01=1%)
                                      [1] : white_threshold, if > dark_threshold will be applied a gradient till white_threshold, range [0-1] (0.01=1%)
                                      [2] : dark_sat, amount of de-saturation to apply to the dark area, range [0-1]
                                      [3] : dark_bright, darkness parameter it used to reduce the "V" component in "HSV" colorspace, range [0, 1]
                                      [4] : "chroma range" parameter (optional), if="none" is disabled (see the README)
    :param colormap:            Direct hue/color mapping (only on ref-frames), without luma filtering, using the "chroma adjustment"
                                parameter, if="none" is disabled.
    :param ex_model:            "Exemplar-based" model to use for the color propagation, available models are:
                                    0 : CMNET2 (default)
                                    1 : Deep-Exemplar
                                    2 : Deep-Remaster
                                    3 : ColorMNet
    :param encode_mode:         Parameter used by CMNET2/ColorMNet to define the encode mode strategy.
                                Available values are:
                                     0: remote encoding. The frames will be colored by a thread outside Vapoursynth.
                                                         This option don't have any GPU memory limitation and will allow
                                                         to fully use the long term frame memory.
                                                         It is the faster encode method (default)
                                     1: local encoding.  The frames will be colored inside the Vapoursynth environment.
                                                         In this case the max_memory will be limited by the size of GPU
                                                         memory (max 15 frames for 24GB GPU).
                                                         Useful for coloring clips with a lot of smooth transitions,
                                                         since in this case is better to use a short frame memory or
                                                         the Deep-Exemplar model, which is faster.
    :param max_memory_frames:   Parameter used by ColorMNet/DeepRemaster models.
                                For ColorMNet specify the max number of encoded frames to keep in memory.
                                Its value depend on encode mode and must be defined manually following the suggested values.
                                encode_mode=0: there is no memory limit (it could be all the frames in the clip).
                                Suggested values are:
                                    min=150, max=10000
                                If = 0 will be filled with the value of 10000 or the clip length if lower.
                                encode_mode=1: the max memory frames is limited by available GPU memory.
                                Suggested values are:
                                    min=1, max=4    : for 8GB GPU
                                    min=1, max=8    : for 12GB GPU
                                    min=1, max=15   : for 24GB GPU
                                If = 0 will be filled with the max value (depending on total GPU RAM available).
                                For DeepRemaster represent the number to reference frames to keep in memory.
                                Suggested values are:
                                    min=2, max=50
                                If = 0 will be filled with the value of 20.
    :param retry_threshold:     (CMNET2 only) Threshold used to identify frames that may benefit from an additional
                                 reference frame. Range [0.0, 1.0], default 0.0 (disabled).
                                 High values (> 0.3) trigger more retry, while lower values (< 0.3) trigger less retry.
                                 Suggested value in the range: 0.20-0.35.
    :param retry_model:         (CMNET2 only) If retry_threshold > 0 it represents the model used to colorize the missing
                                 reference frames. Allowed values, are:
                                     0 : Model HAVC (colorization with: 60%*DeOldify + 40%*DDColor)
                                     1 : Model DiT colorization with model_precision = "fp4" (RTX 50-Series)
                                     2 : Model DiT colorization with model_precision = "int4" (RTX 30/40-Series)
                                 For the models 1 and 2 is necessary to run the DiT Server as explained in the docstring
                                 of HAVC_cmnet2dit(). In the case the DiT Server is not running will be used the model 0
                                 (Model HAVC). Range [0, 1, 2], default = 0
    :param high_resolution:     if true the resolution of the inference will be increased, this will improve the color
                                accuracy, but the inference will be about 2x slower. default = False.
    :param torch_dir:           torch hub dir location, default is model directory, if set to None will switch
                                to torch cache dir
    """
    # disable packages warnings
    disable_warnings()

    if not torch.cuda.is_available():
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: CUDA is not available")

    clip, orig_fmt = convert_format_RGB24(clip)

    if only_ref_frames and (sc_framedir is None):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: only_ref_frames is enabled but sc_framedir is unset")

    if not (sc_framedir is None) and method != 0 and only_ref_frames:
        HAVC_LogMessage(MessageType.EXCEPTION,
                        "HAVC_deepex: only_ref_frames is enabled but method != 0 (HAVC)")

    if method != 0 and (sc_framedir is None):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: method != 0 but sc_framedir is unset")

    if method in (3, 4) and not (clip_ref is None):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: method in (3, 4) but clip_ref is set")

    if method in (0, 1, 2, 5, 6) and (clip_ref is None):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: method in (0, 1, 2, 5, 6) but clip_ref is unset")

    if clip_ref is not None:
        clip_ref, orig_fmt_r = convert_format_RGB24(clip_ref)

    if method not in range(7):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: method must be in range [0-6]")

    if ref_merge not in range(6):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: ref_merge must be in range [0-5]")

    if ref_merge > 0 and (method not in (0, 1, 5) and ex_model != 3):
        HAVC_LogMessage(MessageType.EXCEPTION,
                        "HAVC_deepex: method must be in (0, 1, 5) to be used with ref_merge > 0")

    sc_threshold = None
    sc_frequency = None
    if method in (0, 1, 2):
        sc_threshold, sc_frequency = get_sc_props(clip_ref)
        if sc_threshold == 0 and sc_frequency == 0:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_deepex: method in (0, 1, 2) but sc_threshold and sc_frequency are not set")
        if sc_frequency == 1 and only_ref_frames:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_deepex: only_ref_frames is enabled but sc_frequency == 1 or ColorTemp/FrameInterp are set ")
        if not only_ref_frames and ref_merge > 0 and sc_frequency != 1 and ex_model != 3:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_deepex: method in (0, 1, 2) and ref_merge > 0 but sc_frequency != 1")

    if torch_dir is not None:
        torch.hub.set_dir(torch_dir)

    # ------------------------------------------------------------------------------------------------------------------
    # SPECIAL MANAGEMENT OF DeepRemaster (ex_model = 2)
    if method in (0, 1, 2) and ex_model == 2:
        HAVC_LogMessage(MessageType.EXCEPTION,
                        "HAVC_deepex: DeepRemaster cannot be used with methods: 0, 1, 2 (HAVC)")

    # SPECIAL MANAGEMENT OF METHOD=(5,6)
    if method in (5, 6):
        clip_restored = HAVC_restore_video(clip, clip_ref, method, render_speed, ex_model, ref_merge, ref_weight,
                                           ref_thresh, ref_freq, ref_norm, max_memory_frames, render_vivid, encode_mode)
        return restore_format(clip_restored, orig_fmt)
    # ------------------------------------------------------------------------------------------------------------------

    # creates the directory "sc_framedir" and does not raise an exception if the directory already exists
    if not (sc_framedir is None):
        pathlib.Path(sc_framedir).mkdir(parents=True, exist_ok=True)

    # static params
    enable_resize = high_resolution

    # unpack dark
    dark_enabled = dark
    dark_threshold = dark_p[0]
    dark_amount = dark_p[1]
    if len(dark_p) > 2:
        dark_hue_adjust = dark_p[2]
    else:
        dark_hue_adjust = 'none'

    # unpack chroma_smoothing
    chroma_smoothing_enabled = smooth
    black_threshold = smooth_p[0]
    white_threshold = smooth_p[1]
    dark_sat = smooth_p[2]
    dark_bright = -smooth_p[3]  # change the sign to reduce the bright
    if len(smooth_p) > 4:
        chroma_adjust = smooth_p[4]
    else:
        chroma_adjust = 'none'

    # define colormap
    colormap = colormap.lower()
    colormap_enabled = (colormap != "none" and colormap != "")

    enable_refmerge: bool = (ref_merge > 0 and sc_frequency == 1)
    refmerge_weight: list[float] = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]
    if enable_refmerge:
        if ref_weight is None:
            ref_weight = refmerge_weight[ref_merge]
        if ref_thresh is None:
            ref_thresh = constants.DEF_THRESHOLD
        if ref_freq is None or ref_freq == 1:
            ref_freq = 0
        clip_sc = SceneDetect(clip, threshold=ref_thresh, frequency=ref_freq, frame_norm=ref_norm)
        if method in (1, 2) and not (sc_framedir is None) and not only_ref_frames:
            clip_sc = SceneDetectFromDir(clip_sc, sc_framedir=sc_framedir, merge_ref_frame=True,
                                         ref_frame_ext=(method == 2))
    else:
        ref_weight = 1.0
        clip_sc = None

    if method != 0 and not (sc_framedir is None):
        ref_frame_ext = method in (2, 4)
        merge_ref_frame = method in (1, 2)
        if method in (1, 2):
            clip = SceneDetectFromDir(clip_ref, sc_framedir=sc_framedir, merge_ref_frame=merge_ref_frame,
                                      ref_frame_ext=ref_frame_ext)
            clip_ref = CopySCDetect(clip_ref, clip)
        else:
            clip = SceneDetectFromDir(clip, sc_framedir=sc_framedir, merge_ref_frame=merge_ref_frame,
                                      ref_frame_ext=ref_frame_ext)
    else:
        clip = CopySCDetect(clip, clip_ref)

    clip_orig = clip

    if ex_model == 0:
        d_size = get_render_size(clip.width, clip.height, render_speed=render_speed.lower())
    else:
        d_size = get_deepex_size(render_speed=render_speed.lower(), enable_resize=enable_resize, ex_model=ex_model)
        smc = SmartResizeColorizer(clip_size=d_size, ex_model=ex_model)
        smr = SmartResizeReference(clip_size=d_size, ex_model=ex_model)

    if method != 0 and not (sc_framedir is None):
        if method in (1, 2):
            clip_ref = vs_ext_reference_clip(clip_ref, sc_framedir=sc_framedir, clip_resize=(ex_model == 2))
        else:
            clip_ref = vs_ext_reference_clip(clip, sc_framedir=sc_framedir, clip_resize=(ex_model == 2))

    # clip and clip_ref are resized to match the frame size used for inference
    if ex_model == 0:
        clip = clip.resize.Spline36(width=d_size[0], height=d_size[1])
        clip_ref = clip_ref.resize.Spline36(width=d_size[0], height=d_size[1])
    else:
        clip = smc.get_resized_clip(clip)
        clip_ref = smr.get_resized_clip(clip_ref)

    if colormap_enabled:
        clip_ref = vs_sc_colormap(clip_ref, colormap=colormap)

    if dark_enabled:
        clip_ref = vs_sc_dark_tweak(clip_ref, dark_threshold=dark_threshold, dark_amount=dark_amount,
                                    dark_hue_adjust=dark_hue_adjust.lower())

    if chroma_smoothing_enabled:
        clip_ref = vs_sc_chroma_bright_tweak(clip_ref, black_threshold=black_threshold, white_threshold=white_threshold,
                                             dark_sat=dark_sat, dark_bright=dark_bright,
                                             chroma_adjust=chroma_adjust.lower())
    ref_same_as_video = method == 3  # unico caso in cui è True il flag

    if ex_model == 3 and max_memory_frames > 1:
        render_vivid = False

    if only_ref_frames:
        clip_colored = clip_ref
    else:
        match ex_model:
            case 0:  # CMNET2
                clip_colored = vs_colormnet2(clip, clip_ref, clip_sc, image_size=-1, enable_resize=enable_resize,
                                            encode_mode=encode_mode, max_memory_frames=max_memory_frames,
                                            frame_propagate=ref_same_as_video, render_vivid=render_vivid,
                                            ref_weight=ref_weight, retry_perm_share_threshold=retry_threshold, retry_model=retry_model)
            case 1:  # Deep-Exemplar
                clip_colored = vs_deepex(clip, clip_ref, clip_sc, image_size=d_size, enable_resize=enable_resize,
                                         propagate=ref_same_as_video, wls_filter_on=True, render_vivid=render_vivid,
                                         ref_weight=ref_weight)
            case 2:  # DeepRemaster
                clip_colored = vs_deepremaster(clip, clip_ref, clip_sc, render_vivid=render_vivid,
                                               ref_weight=ref_weight, memory_size=max_memory_frames)
            case 3:  # ColorMNet
                clip_colored = vs_colormnet(clip, clip_ref, clip_sc, image_size=-1, enable_resize=enable_resize,
                                            encode_mode=encode_mode, max_memory_frames=max_memory_frames,
                                            frame_propagate=ref_same_as_video, render_vivid=render_vivid,
                                            ref_weight=ref_weight)
            case _:
                clip_colored = None
                HAVC_LogMessage(MessageType.EXCEPTION, "HybridAVC: unknown exemplar model id: " + str(ex_model))

    if ex_model == 0:
        clip_resized = clip_colored.resize.Spline36(width=clip_orig.width, height=clip_orig.height)
    else:
        clip_resized = smc.restore_clip_size(clip_colored)

    # restore original resolution details, 5% faster than ShufflePlanes()
    if not (sc_framedir is None) and method == 0 and only_ref_frames:
        # ref frames are saved if sc_framedir is set
        clip_new = vs_sc_recover_clip_luma(clip_orig, clip_resized, scenechange=True, sc_framedir=sc_framedir)
    else:
        clip_new =  vs_recover_clip_luma(clip_orig, clip_resized)

    return restore_format(clip_new, orig_fmt)

def HAVC_cmnet2(clip: vs.VideoNode = None, clip_ref: vs.VideoNode = None, method: int = 0, render_speed: str = 'auto',
                render_vivid: bool = False, ref_merge: int = 0, sc_framedir: str = None, ref_norm: bool = False, dark: bool = False,
                dark_p: list = (0.2, 0.8), smooth: bool = False, smooth_p: list = (0.3, 0.7, 0.9, 0.0, "none"),
                colormap: str = "none", ref_weight: float = None, ref_thresh: float = None, ref_freq: int = None,
                encode_mode: int = 0, max_memory_frames: int = 0, ref_mode: int = 1, retry_threshold: float = 0.0,
                retry_model: int = 0, torch_dir: str = model_dir) -> vs.VideoNode:
    """CMNET2 colorization filter

    :param clip:                Clip to process, any clip format is supported
    :param clip_ref:            Clip containing the reference frames (necessary if method=0,1,2,5,6)
    :param method:              Method to use to generate reference frames (RF).
                                        0 = HAVC same as video (default)
                                        1 = HAVC + RF same as video
                                        2 = HAVC + RF different from video
                                        3 = external RF same as video
                                        4 = external RF different from video
                                        5 = external ClipRef same as video
                                        6 = external ClipRef different from video
    :param render_speed:        Preset to control the render method and speed:
                                Allowed values are:
                                        'Auto'   : will be automatically assigned the optimal render size (default)
                                        'Fast'   : colors are more washed out
                                        'Medium' : colors are a little washed out
                                        'Slow'   : colors are a little more vivid
                                        'Slower' : colors are more accurate (usually is very slow)
    :param render_vivid:        If True, the saturation will be increased by about 15%. Default: False
    :param ref_merge:           Method used by DeepEx to merge the reference frames with the frames propagated by DeepEx.
                                It is applicable only with DeepEx method: 0, 1, 5.
                                The HAVC reference frames must be produced with frequency = 1.
                                Allowed values are:
                                        0 = No RF merge (reference frames can be produced with any frequency)
                                        1 = RF-Merge VeryLow (reference frames are merged with weight=0.3)
                                        2 = RF-Merge Low (reference frames are merged with weight=0.4)
                                        3 = RF-Merge Med (reference frames are merged with weight=0.5)
                                        4 = RF-Merge High (reference frames are merged with weight=0.6)
                                        5 = RF-Merge VeryHigh (reference frames are merged with weight=0.7)
    :param ref_weight:          If (ref_merge > 0), represent the weight used to merge the reference frames.
                                If is not set, is assigned automatically a value depending on ref_merge/method values.
    :param ref_thresh:          Represent the threshold used to create the reference frames. If is not set, is assigned
                                automatically a value of 0.10
    :param ref_freq:            If > 0 will be generated at least a reference frame every "ref_freq" frames.
                                range [0-1500]. If is not set, is assigned automatically a value depending on
                                ref_merge/method values.
    :param ref_norm:            If true the B&W frames are normalized before apply the Scene Detection to generate the
                                reference frames. The normalization will increase the sensitivity to smooth scene changes,
                                range [True, False], default: False
    :param sc_framedir:         If set, define the directory where are stored the reference frames. If only_ref_frames=True,
                                and method=0 this directory will be written with the reference frames used by the filter.
                                if method!=0 the directory will be read to create the reference frames that will be used
                                by "Exemplar-based" Video Colorization. The reference frame name must be in the
                                format: ref_nnnnnn.[jpg|png], for example the reference frame 897 must be
                                named: ref_000897.png. With methods 5,6 this parameters can be the path to a video clip.
                                NOTE: When used with method in (1, 2, 3, 4), reference frames are read directly
                                      from this directory instead of being re-evaluated through the VapourSynth
                                      pipeline for each preload/slide operation. This is significantly faster
                                      but means that reference-frame filters (colormap, dark, smooth) are NOT
                                      applied to the permanent-memory refs — they are applied only to the VS
                                      clip_ref used for the runtime merge. If you need those filters applied
                                      uniformly, apply equivalent post-processing to the final colored clip.
    :param dark:                Enable/disable darkness filter (only on ref-frames), range [True,False]
    :param dark_p:              Parameters for darken the clip's dark portions, which sometimes are wrongly colored by the color models
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0.1-0.5] (0.01=1%)
                                      [1] : dark_amount: amount of desaturation to apply to the dark area, range [0-1]
                                      [2] : "chroma range" parameter (optional), if="none" is disabled (see the README)
    :param smooth:              Enable/disable chroma smoothing (only on ref-frames), range [True, False]
    :param smooth_p:            parameters to adjust the saturation and "vibrancy" of the clip.
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0-1] (0.01=1%)
                                      [1] : white_threshold, if > dark_threshold will be applied a gradient till white_threshold, range [0-1] (0.01=1%)
                                      [2] : dark_sat, amount of de-saturation to apply to the dark area, range [0-1]
                                      [3] : dark_bright, darkness parameter it used to reduce the "V" component in "HSV" colorspace, range [0, 1]
                                      [4] : "chroma range" parameter (optional), if="none" is disabled (see the README)
    :param colormap:            Direct hue/color mapping (only on ref-frames), without luma filtering, using the "chroma adjustment"
                                parameter, if="none" is disabled.
    :param encode_mode:         Parameter used by ColorMNet to define the encode mode strategy.
                                Available values are:
                                     0: remote encoding. The frames will be colored by a thread outside Vapoursynth.
                                                         This option has no GPU memory limitation and fully exploits
                                                         the long-term frame memory. It is the faster encode method
                                                         (default). All available reference frames are used via the
                                                         sliding permanent-memory window.
                                     1: local encoding.  The frames will be colored inside the Vapoursynth environment.
                                                         Useful when remote encoding is not available or when a single
                                                         process is preferred.
    :param max_memory_frames:   Window size for the sliding permanent-memory of ColorMNet2.
                                Defines how many reference frames are held in the model's permanent memory at any
                                given time. Must be an even number and must not exceed the number of available
                                reference frames. The window slides forward automatically as colorization advances.
                                Suggested values: min=10, max=500.
                                If = 0 (default) will be set to DEF_XRF_WINDOW_SIZE (20).
    :param ref_mode:            Mode selected to access to the external reference frames.
                                Allowed values are:
                                    0: will use direct access to reference frame folder
                                    1: will use Vapoursynth clips to access to reference frames (default)
    :param retry_threshold:     Threshold used to identify frames that may benefit from an additional
                                reference frame. Range [0.0, 1.0], default 0.0 (disabled).
                                High values (> 0.3) trigger more retry, while lower values (< 0.3) trigger less retry.
                                Suggested value in the range: 0.20-0.35.
    :param retry_model:         If retry_threshold > 0 it represents the model used to colorize the missing
                                reference frames. Allowed values, are:
                                     0 : Model HAVC (colorization with: 60%*DeOldify + 40%*DDColor)
                                     1 : Model DiT colorization with model_precision = "fp4" (RTX 50-Series)
                                     2 : Model DiT colorization with model_precision = "int4" (RTX 30/40-Series)
                                For the models 1 and 2 is necessary to run the DiT Server as explained in the docstring
                                of HAVC_cmnet2dit(). In the case the DiT Server is not running will be used the model 0
                                (Model HAVC). Range [0, 1, 2], default = 0
    :param torch_dir:           torch hub dir location, default is model directory, if set to None will switch
                                to torch cache dir
    """
    # disable packages warnings
    disable_warnings()

    # static variables
    only_ref_frames: bool = False

    if not torch.cuda.is_available():
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_cmnet2: CUDA is not available")

    clip, orig_fmt = convert_format_RGB24(clip)

    if clip_ref is not None:
        clip_ref, orig_fmt_r = convert_format_RGB24(clip_ref)

    if method not in range(7):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_cmnet2: method must be in range [0-6]")

    if ref_merge not in range(6):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_cmnet2: ref_merge must be in range [0-5]")

    sc_threshold = None
    sc_frequency = None
    if method in (0, 1, 2):
        sc_threshold, sc_frequency = get_sc_props(clip_ref)
        if sc_threshold == 0 and sc_frequency == 0:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_deepex: method in (0, 1, 2) but sc_threshold and sc_frequency are not set")
        if sc_frequency == 1 and only_ref_frames:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_deepex: only_ref_frames is enabled but sc_frequency == 1")
        if not only_ref_frames and ref_merge > 0 and sc_frequency != 1:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_deepex: method in (0, 1, 2) and ref_merge > 0 but sc_frequency != 1")

    if torch_dir is not None:
        torch.hub.set_dir(torch_dir)

    # static params
    enable_resize = False

    # unpack dark
    dark_enabled = dark
    dark_threshold = dark_p[0]
    dark_amount = dark_p[1]
    if len(dark_p) > 2:
        dark_hue_adjust = dark_p[2]
    else:
        dark_hue_adjust = 'none'

    # unpack chroma_smoothing
    chroma_smoothing_enabled = smooth
    black_threshold = smooth_p[0]
    white_threshold = smooth_p[1]
    dark_sat = smooth_p[2]
    dark_bright = -smooth_p[3]  # change the sign to reduce the bright
    if len(smooth_p) > 4:
        chroma_adjust = smooth_p[4]
    else:
        chroma_adjust = 'none'

    # define colormap
    colormap = colormap.lower()
    colormap_enabled = (colormap != "none" and colormap != "")

    enable_refmerge: bool = (ref_merge > 0 and sc_frequency == 1)
    refmerge_weight: list[float] = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
    if enable_refmerge:
        if ref_weight is None:
            ref_weight = refmerge_weight[ref_merge]
        if ref_thresh is None:
            ref_thresh = constants.DEF_THRESHOLD
        if ref_freq is None or ref_freq == 1:
            ref_freq = 0
        clip_sc = SceneDetect(clip, threshold=ref_thresh, frequency=ref_freq, frame_norm=ref_norm)
        if method in (1, 2) and not (sc_framedir is None) and not only_ref_frames:
            clip_sc = SceneDetectFromDir(clip_sc, sc_framedir=sc_framedir, merge_ref_frame=True,
                                         ref_frame_ext=(method == 2))
    else:
        ref_weight = 1.0
        clip_sc = None

    if method != 0 and not (sc_framedir is None):
        ref_frame_ext = method in (2, 4)
        merge_ref_frame = method in (1, 2)
        if method in (1, 2):
            clip = SceneDetectFromDir(clip_ref, sc_framedir=sc_framedir, merge_ref_frame=merge_ref_frame,
                                      ref_frame_ext=ref_frame_ext)
            clip_ref = CopySCDetect(clip_ref, clip)
        else:
            clip = SceneDetectFromDir(clip, sc_framedir=sc_framedir, merge_ref_frame=merge_ref_frame,
                                      ref_frame_ext=ref_frame_ext)
    else:
        clip = CopySCDetect(clip, clip_ref)

    clip_orig = clip

    # when reference frames exist on disk, read them directly (bypasses VS pipeline
    # re-evaluation per reference — significant speedup on preload/slide).
    # The clip_ref VS path is still used for the merge at runtime (1 get_frame per
    # output frame, not per reference).
    use_dir_refs = (ref_mode == 0
                    and method in (1, 2, 3, 4)
                    and sc_framedir is not None
                    and os.path.isdir(sc_framedir))

    # if user explicitly requested ref_mode=0 but the conditions aren't met,
    # warn and fall back to VS mode silently — alternative: raise exception
    if ref_mode == 0 and not use_dir_refs:
        HAVC_LogMessage(MessageType.WARNING,
                        "HAVC_cmnet2: ref_mode=0 (direct) requested but not applicable "
                        "(requires method in (1,2,3,4) and valid sc_framedir). "
                        "Falling back to VS clip mode.")

    if method != 0 and not (sc_framedir is None):
        if method in (1, 2):
            clip_ref = vs_ext_reference_clip(clip_ref, sc_framedir=sc_framedir)
        else:
            clip_ref = vs_ext_reference_clip(clip, sc_framedir=sc_framedir)

    d_size = get_render_size(clip.width, clip.height, render_speed=render_speed.lower())
    clip = clip.resize.Spline36(width=d_size[0], height=d_size[1])
    clip_ref = clip_ref.resize.Spline36(width=d_size[0], height=d_size[1])

    # when reference frames are loaded directly from disk, filters on clip_ref
    # (colormap, dark, smooth) are redundant: perm_mem refs bypass them entirely,
    # and applying them only to the runtime merge clip creates an inconsistency
    # that degrades merge quality. Skip them.
    if use_dir_refs:
        if colormap_enabled or dark_enabled or chroma_smoothing_enabled:
            HAVC_LogMessage(MessageType.WARNING,
                            "HAVC_cmnet2: ref-frame filters (colormap/dark/smooth) "
                            "are ignored in ref_mode=0. Apply equivalent post-"
                            "processing to the colored clip if needed.")
        colormap_enabled = False
        dark_enabled = False
        chroma_smoothing_enabled = False

    if colormap_enabled:
        clip_ref = vs_sc_colormap(clip_ref, colormap=colormap)

    if dark_enabled:
        clip_ref = vs_sc_dark_tweak(clip_ref, dark_threshold=dark_threshold, dark_amount=dark_amount,
                                    dark_hue_adjust=dark_hue_adjust.lower())

    if chroma_smoothing_enabled:
        clip_ref = vs_sc_chroma_bright_tweak(clip_ref, black_threshold=black_threshold, white_threshold=white_threshold,
                                             dark_sat=dark_sat, dark_bright=dark_bright,
                                             chroma_adjust=chroma_adjust.lower())
    ref_same_as_video = False

    clip_colored = vs_colormnet2(clip, clip_ref, clip_sc, image_size=-1, enable_resize=enable_resize,
                                            encode_mode=encode_mode, max_memory_frames=max_memory_frames,
                                            frame_propagate=ref_same_as_video, render_vivid=render_vivid,
                                            ref_weight=ref_weight, sc_framedir=sc_framedir if use_dir_refs else None,
                                            retry_perm_share_threshold=retry_threshold, retry_model=retry_model)

    clip_resized = clip_colored.resize.Spline36(width=clip_orig.width, height=clip_orig.height)

    # restore original resolution details, 5% faster than ShufflePlanes()
    clip_new = vs_recover_clip_luma(clip_orig, clip_resized)

    return restore_format(clip_new, orig_fmt)
    

def HAVC_cmnet2dit(clip: vs.VideoNode = None,
                   render_speed: str = 'auto',
                   render_vivid: bool = False,
                   sc_thresh: float = 0.035,
                   sc_tht_ssim: float= 0.80,
                   sc_min_int: int = 25,
                   sc_tht_offset: int = 2,
                   sc_min_freq: int = 0,
                   max_memory_frames: int = 20,
                   dit_engine_params: dict = None,
                   retry_threshold: float = 0.0,
                   retry_model: int = 1,
                   torch_dir: str = model_dir) -> vs.VideoNode:
    """CMNET2-DIT colorization filter.

    Like HAVC_cmnet2() but designed for B&W reference frames: scene-change
    frames extracted from the input clip are colorized by a DiT-based model
    (DiT Engine, accessed via RPC) *before* being loaded into CMNET2
    permanent memory.  This makes HAVC_cmnet2dit() self-contained, no
    eparate pre-colored reference clip is needed.

    The DiT colorization of reference frames always runs in pairs
    (colorize_image_pair()) to exploit the DiT model's batched forward pass.
    A single colorize_image() call handles any odd leftover reference frame at
    the end of the clip.

    :param clip:                B&W source clip; any format is accepted and
                                converted internally to RGB24.
    :param render_speed:        Preset controlling CMNET2 render resolution.
                                Allowed values (case-insensitive):
                                    'Auto'   – optimal size chosen automatically (default)
                                    'Fast'   – more washed-out colours
                                    'Medium' – slightly washed out
                                    'Slow'   – slightly more vivid
                                    'Slower' – most accurate (usually very slow)
    :param render_vivid:        If True, apply a ~15% saturation boost after
                                colorization. Default: False.
    :param sc_thresh:           Scene edges-detection threshold used to select
                                reference frames from the input clip.
                                Range [0.01, 0.15]. Default: 0.035.
    :param sc_tht_ssim:         Threshold used by the SSIM (Structural Similarity Index Metric) selection filter.
                                If > 0, will be activated a filter that will improve the scene-change detection,
                                by discarding images that are similar. Suggested values are between 0.35 and 0.85,
                                range [0-1], default 0.80
    :param sc_min_int:          Minimum frame distance between scene changes. Default 25.
    :param sc_tht_offset:       Offset index used for the Scene change detection. The comparison will be performed,
                                between frame[n] and frame[n-offset]. An offset > 1 is useful to detect blended scene
                                change, range[1, 25]. Default = 2.
    :param sc_min_freq:         If > 0 will be generated at least 1 reference frame every "sc_min_freq" frames.
                                range [0-1000], default: 0.
    :param max_memory_frames:   Sliding permanent-memory window size for CMNET2.
                                Automatically rounded down to the nearest even number
                                (required for pair-wise DIT colorization).
                                0 → DEF_XRF_WINDOW_SIZE (20). Suggested: 10–500. Default = 20
    :param dit_engine_params:   Optional dict of keyword arguments forwarded to
                                DiT Engine Server.  Any key not provided falls back
                                to the DiT Engine Server default.  Recognized keys:
                                    host            : RPC server address (default "127.0.0.1")
                                    port            : RPC server port   (default 8765)
                                    model_name      : Nunchaku model name
                                    model_precision : "fp4" (RTX 50xx) or "int4" (RTX 30/40xx)
                                    model_rank      : SVD rank "32" or "128"
                                    model_inference_steps : steps used to pick the model file
                                    cache_dir       : HuggingFace cache directory
                                    full_model_path : absolute path to a local .safetensors
                                    prompt          : text prompt guiding colorization
                                    steps           : inference steps per image (default 2)
                                    img_size        : max long-side in pixels before inference (0 = original size)
    :param retry_threshold:     Threshold used to identify frames that may benefit from
                                 an additional reference frame. Range [0.0, 1.0],
                                 default 0.0 (disabled). Suggested: 0.20-0.35.
    :param retry_model:         If retry_threshold > 0, model used to colorize missing (default: 1)
                                reference frames. Allowed values are:
                                     0 = HAVC (DeOldify + DDColor),
                                     1 = DiT fp4,
                                     2 = DiT int4.
    :param torch_dir:           Torch hub directory for CMNET2 model weights.
                                Default: package model directory.
                                Pass None to use the Torch cache directory.
    :return:                    Colorized clip in the same format as the input.
    """

    disable_warnings()

    if not torch.cuda.is_available():
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_cmnet2dit: CUDA is not available")

    clip, orig_fmt = convert_format_RGB24(clip)

    if torch_dir is not None:
        torch.hub.set_dir(torch_dir)

    # -----------------------------------------------------------------------
    # Scene-change detection — produces ref frames from the B&W input clip
    # -----------------------------------------------------------------------
    # Apply defaults when caller did not provide explicit thresholds.
    if sc_thresh is None:
        ref_thresh = 0.035
    if sc_min_int is None:
        ref_freq = 25

    # forced encode_mode=0 due to memory limitation
    encode_mode = 0

    # Run scene detection on the (resized-later, but prop-only here) clip to
    # obtain _SceneChangePrev props.  clip_ref == clip internally: the B&W
    # input clip is both the content to colorize and the source of reference
    # frames that DiT Engine Server will colorize.
    clip_ref = SceneDetectEdges(clip, threshold=sc_thresh, frequency=sc_min_freq, ssim_threshold=sc_tht_ssim,
                                sc_diff_offset=sc_tht_offset, sc_min_int=sc_min_int, sc_mult_tht=15,
                                tht_white=0.70, tht_black=0.10)
    # Copy scene-change props to the working clip so that downstream VS filters
    # (e.g. vs_recover_clip_luma) can access them if needed.
    clip = CopySCDetect(clip, clip_ref)

    clip_orig = clip

    # No ref-merge in the DIT path.
    ref_same_as_video = False
    clip_sc = None

    # -----------------------------------------------------------------------
    # Resize to model inference resolution
    # -----------------------------------------------------------------------
    enable_resize = False   # static: consistent with HAVC_cmnet2 default

    d_size = get_render_size(clip.width, clip.height, render_speed=render_speed.lower())
    clip = clip.resize.Spline36(width=d_size[0], height=d_size[1])
    clip_ref = clip_ref.resize.Spline36(width=d_size[0], height=d_size[1])

    # -----------------------------------------------------------------------
    # CMNET2-DIT colorization
    # -----------------------------------------------------------------------
    clip_colored = vs_colormnet2dit(
        clip, clip_ref,
        dit_engine_params=dit_engine_params,
        image_size=-1,
        enable_resize=enable_resize,
        encode_mode=encode_mode,
        max_memory_frames=max_memory_frames,
        frame_propagate=ref_same_as_video,
        render_vivid=render_vivid,
        retry_perm_share_threshold=retry_threshold,
        retry_model=retry_model,
    )

    # -----------------------------------------------------------------------
    # Restore original resolution and format
    # -----------------------------------------------------------------------
    clip_resized = clip_colored.resize.Spline36(width=clip_orig.width, height=clip_orig.height)

    # Graft the original luma back onto the coloured chroma for sharpness
    # (~5% faster than ShufflePlanes).
    clip_new = vs_recover_clip_luma(clip_orig, clip_resized)

    return restore_format(clip_new, orig_fmt)


def HAVC_cmnet1(clip: vs.VideoNode = None, clip_ref: vs.VideoNode = None, method: int = 0, render_speed: str = 'auto',
                render_vivid: bool = True, ref_merge: int = 0, sc_framedir: str = None, ref_norm: bool = False, dark: bool = False,
                dark_p: list = (0.2, 0.8), smooth: bool = False, smooth_p: list = (0.3, 0.7, 0.9, 0.0, "none"),
                colormap: str = "none", ref_weight: float = None, ref_thresh: float = None, ref_freq: int = None,
                encode_mode: int = 0, max_memory_frames: int = 0, torch_dir: str = model_dir) -> vs.VideoNode:
    """ColorMNet colorization filter

    :param clip:                Clip to process, any clip format is supported
    :param clip_ref:            Clip containing the reference frames (necessary if method=0,1,2,5,6)
    :param method:              Method to use to generate reference frames (RF).
                                        0 = HAVC same as video (default)
                                        1 = HAVC + RF same as video
                                        2 = HAVC + RF different from video
                                        3 = external RF same as video
                                        4 = external RF different from video
                                        5 = external ClipRef same as video
                                        6 = external ClipRef different from video
    :param render_speed:        Preset to control the render method and speed:
                                Allowed values are:
                                        'Auto'   : will be automatically assigned the optimal render size (default)
                                        'Fast'   : colors are more washed out
                                        'Medium' : colors are a little washed out
                                        'Slow'   : colors are a little more vivid
                                        'Slower' : colors are more accurate (usually is very slow)
    :param render_vivid:        If True, the frames memory is reset at every reference frame update. Default: True
    :param ref_merge:           Method used by DeepEx to merge the reference frames with the frames propagated by DeepEx.
                                It is applicable only with DeepEx method: 0, 1, 5.
                                The HAVC reference frames must be produced with frequency = 1.
                                Allowed values are:
                                        0 = No RF merge (reference frames can be produced with any frequency)
                                        1 = RF-Merge VeryLow (reference frames are merged with weight=0.3)
                                        2 = RF-Merge Low (reference frames are merged with weight=0.4)
                                        3 = RF-Merge Med (reference frames are merged with weight=0.5)
                                        4 = RF-Merge High (reference frames are merged with weight=0.6)
                                        5 = RF-Merge VeryHigh (reference frames are merged with weight=0.7)
    :param ref_weight:          If (ref_merge > 0), represent the weight used to merge the reference frames.
                                If is not set, is assigned automatically a value depending on ref_merge/method values.
    :param ref_thresh:          Represent the threshold used to create the reference frames. If is not set, is assigned
                                automatically a value of 0.10
    :param ref_freq:            If > 0 will be generated at least a reference frame every "ref_freq" frames.
                                range [0-1500]. If is not set, is assigned automatically a value depending on
                                ref_merge/method values.
    :param ref_norm:            If true the B&W frames are normalized before apply the Scene Detection to generate the
                                reference frames. The normalization will increase the sensitivity to smooth scene changes,
                                range [True, False], default: False
    :param sc_framedir:         If set, define the directory where are stored the reference frames. If only_ref_frames=True,
                                and method=0 this directory will be written with the reference frames used by the filter.
                                if method!=0 the directory will be read to create the reference frames that will be used
                                by "Exemplar-based" Video Colorization. The reference frame name must be in the
                                format: ref_nnnnnn.[jpg|png], for example the reference frame 897 must be
                                named: ref_000897.png. With methods 5,6 this parameters can be the path to a video clip.
                                NOTE: When used with method in (1, 2, 3, 4), reference frames are read directly
                                      from this directory instead of being re-evaluated through the VapourSynth
                                      pipeline for each preload/slide operation. This is significantly faster
                                      but means that reference-frame filters (colormap, dark, smooth) are NOT
                                      applied to the permanent-memory refs — they are applied only to the VS
                                      clip_ref used for the runtime merge. If you need those filters applied
                                      uniformly, apply equivalent post-processing to the final colored clip.
    :param dark:                Enable/disable darkness filter (only on ref-frames), range [True,False]
    :param dark_p:              Parameters for darken the clip's dark portions, which sometimes are wrongly colored by the color models
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0.1-0.5] (0.01=1%)
                                      [1] : dark_amount: amount of desaturation to apply to the dark area, range [0-1]
                                      [2] : "chroma range" parameter (optional), if="none" is disabled (see the README)
    :param smooth:              Enable/disable chroma smoothing (only on ref-frames), range [True, False]
    :param smooth_p:            parameters to adjust the saturation and "vibrancy" of the clip.
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0-1] (0.01=1%)
                                      [1] : white_threshold, if > dark_threshold will be applied a gradient till white_threshold, range [0-1] (0.01=1%)
                                      [2] : dark_sat, amount of de-saturation to apply to the dark area, range [0-1]
                                      [3] : dark_bright, darkness parameter it used to reduce the "V" component in "HSV" colorspace, range [0, 1]
                                      [4] : "chroma range" parameter (optional), if="none" is disabled (see the README)
    :param colormap:            Direct hue/color mapping (only on ref-frames), without luma filtering, using the "chroma adjustment"
                                parameter, if="none" is disabled.
    :param encode_mode:         Parameter used by ColorMNet to define the encode mode strategy.
                                Available values are:
                                     0: remote encoding. The frames will be colored by a thread outside Vapoursynth.
                                                         This option has no GPU memory limitation and fully exploits
                                                         the long-term frame memory. It is the faster encode method
                                                         (default). All available reference frames are used via the
                                                         sliding permanent-memory window.
                                     1: local encoding.  The frames will be colored inside the Vapoursynth environment.
                                                         Useful when remote encoding is not available or when a single
                                                         process is preferred.
    :param max_memory_frames:   Defines how many reference frames are held in the model's memory at any
                                given time. Must not exceed the number of available reference frames.
                                Suggested values: min=0, max=5000.
                                If = 0 (default) will be set to 5000.
    :param torch_dir:           torch hub dir location, default is model directory, if set to None will switch
                                to torch cache dir
    """
    # disable packages warnings
    disable_warnings()

    # static variables
    only_ref_frames: bool = False

    if not torch.cuda.is_available():
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_cmnet1: CUDA is not available")

    clip, orig_fmt = convert_format_RGB24(clip)

    if clip_ref is not None:
        clip_ref, orig_fmt_r = convert_format_RGB24(clip_ref)

    if method not in range(7):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_cmnet1: method must be in range [0-6]")

    if ref_merge not in range(6):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_cmnet1: ref_merge must be in range [0-5]")

    sc_threshold = None
    sc_frequency = None
    if method in (0, 1, 2):
        sc_threshold, sc_frequency = get_sc_props(clip_ref)
        if sc_threshold == 0 and sc_frequency == 0:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_cmnet1: method in (0, 1, 2) but sc_threshold and sc_frequency are not set")
        if sc_frequency == 1 and only_ref_frames:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_cmnet1: only_ref_frames is enabled but sc_frequency == 1")
        if not only_ref_frames and ref_merge > 0 and sc_frequency != 1:
            HAVC_LogMessage(MessageType.EXCEPTION,
                            "HAVC_cmnet1: method in (0, 1, 2) and ref_merge > 0 but sc_frequency != 1")

    if torch_dir is not None:
        torch.hub.set_dir(torch_dir)

    # static params
    enable_resize = False

    # unpack dark
    dark_enabled = dark
    dark_threshold = dark_p[0]
    dark_amount = dark_p[1]
    if len(dark_p) > 2:
        dark_hue_adjust = dark_p[2]
    else:
        dark_hue_adjust = 'none'

    # unpack chroma_smoothing
    chroma_smoothing_enabled = smooth
    black_threshold = smooth_p[0]
    white_threshold = smooth_p[1]
    dark_sat = smooth_p[2]
    dark_bright = -smooth_p[3]  # change the sign to reduce the bright
    if len(smooth_p) > 4:
        chroma_adjust = smooth_p[4]
    else:
        chroma_adjust = 'none'

    # define colormap
    colormap = colormap.lower()
    colormap_enabled = (colormap != "none" and colormap != "")

    enable_refmerge: bool = (ref_merge > 0 and sc_frequency == 1)
    refmerge_weight: list[float] = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
    if enable_refmerge:
        if ref_weight is None:
            ref_weight = refmerge_weight[ref_merge]
        if ref_thresh is None:
            ref_thresh = constants.DEF_THRESHOLD
        if ref_freq is None or ref_freq == 1:
            ref_freq = 0
        clip_sc = SceneDetect(clip, threshold=ref_thresh, frequency=ref_freq, frame_norm=ref_norm)
        if method in (1, 2) and not (sc_framedir is None) and not only_ref_frames:
            clip_sc = SceneDetectFromDir(clip_sc, sc_framedir=sc_framedir, merge_ref_frame=True,
                                         ref_frame_ext=(method == 2))
    else:
        ref_weight = 1.0
        clip_sc = None

    if method != 0 and not (sc_framedir is None):
        ref_frame_ext = method in (2, 4)
        merge_ref_frame = method in (1, 2)
        if method in (1, 2):
            clip = SceneDetectFromDir(clip_ref, sc_framedir=sc_framedir, merge_ref_frame=merge_ref_frame,
                                      ref_frame_ext=ref_frame_ext)
            clip_ref = CopySCDetect(clip_ref, clip)
        else:
            clip = SceneDetectFromDir(clip, sc_framedir=sc_framedir, merge_ref_frame=merge_ref_frame,
                                      ref_frame_ext=ref_frame_ext)
    else:
        clip = CopySCDetect(clip, clip_ref)

    clip_orig = clip

    if method != 0 and not (sc_framedir is None):
        if method in (1, 2):
            clip_ref = vs_ext_reference_clip(clip_ref, sc_framedir=sc_framedir)
        else:
            clip_ref = vs_ext_reference_clip(clip, sc_framedir=sc_framedir)

    d_size = get_render_size(clip.width, clip.height, render_speed=render_speed.lower())
    clip = clip.resize.Spline36(width=d_size[0], height=d_size[1])
    clip_ref = clip_ref.resize.Spline36(width=d_size[0], height=d_size[1])

    if colormap_enabled:
        clip_ref = vs_sc_colormap(clip_ref, colormap=colormap)

    if dark_enabled:
        clip_ref = vs_sc_dark_tweak(clip_ref, dark_threshold=dark_threshold, dark_amount=dark_amount,
                                    dark_hue_adjust=dark_hue_adjust.lower())

    if chroma_smoothing_enabled:
        clip_ref = vs_sc_chroma_bright_tweak(clip_ref, black_threshold=black_threshold, white_threshold=white_threshold,
                                             dark_sat=dark_sat, dark_bright=dark_bright,
                                             chroma_adjust=chroma_adjust.lower())
    ref_same_as_video = False

    clip_colored = vs_colormnet(clip, clip_ref, clip_sc, image_size=-1, enable_resize=enable_resize,
                                            encode_mode=encode_mode, max_memory_frames=max_memory_frames,
                                            frame_propagate=ref_same_as_video, render_vivid=render_vivid,
                                            ref_weight=ref_weight)

    clip_resized = clip_colored.resize.Spline36(width=clip_orig.width, height=clip_orig.height)

    # restore original resolution details, 5% faster than ShufflePlanes()
    clip_new = vs_recover_clip_luma(clip_orig, clip_resized)

    return restore_format(clip_new, orig_fmt)

def HAVC_restore_video(clip: vs.VideoNode = None, clip_ref: vs.VideoNode = None, method: int = 6,
                       render_speed: str = 'auto', ex_model: int = 0, ref_merge: int = 0, ref_weight: float = None,
                       ref_thresh: float = None, ref_freq: int = None, ref_norm: bool = False,
                       max_memory_frames: int = 0, render_vivid: bool = False, encode_mode: int = 0,
                       retry_threshold: float = 0, retry_model: int = 0, torch_dir: str = model_dir) -> vs.VideoNode:
    """Colorization Function using DeepRemaster/ColorMNet to restore external video provided externally in clip_ref

    :param clip:                Clip to process, any format is supported
    :param clip_ref:            Clip containing the reference frames (necessary if method=5 or 6)
    :param render_speed:        Preset to control the render method and speed:
                                Allowed values are:
                                        'Auto'   : will be automatically assigned the optimal render size (default)
                                        'Fast'   : colors are more washed out
                                        'Medium' : colors are a little washed out
                                        'Slow'   : colors are a little more vivid
                                        'Slower' : colors are more accurate (usually is very slow)
    :param ex_model:            "Exemplar-based" model to use for the color propagation, available models are:
                                        0 : CMNET2 (default)
                                        1 : Deep-Exemplar
                                        2 : Deep-Remaster
                                        3 : ColorMNet
    :param method:              Method to use to generate reference frames (RF) for the merge.
                                        5 = HAVC restore same as video
                                        6 = HAVC restore different from video (default)
    :param ref_merge:          Method used by DeepEx to merge the reference frames with the frames propagated by DeepEx.
                                It is applicable only with DeepEx method: 0, 1, 5.
                                The HAVC reference frames must be produced with frequency = 1.
                                Allowed values are:
                                        0 = No RF merge (reference frames can be produced with any frequency)
                                        1 = RF-Merge VeryLow (reference frames are merged with weight=0.3)
                                        2 = RF-Merge Low (reference frames are merged with weight=0.4)
                                        3 = RF-Merge Med (reference frames are merged with weight=0.5)
                                        4 = RF-Merge High (reference frames are merged with weight=0.6)
                                        5 = RF-Merge VeryHigh (reference frames are merged with weight=0.7)
    :param ref_weight:          If (ref_merge > 0), represent the weight used to merge the reference frames.
                                If is not set, is assigned automatically a value depending on ref_merge/method value.
    :param ref_thresh:          Represent the threshold used to create the reference frames. If is not set, is assigned
                                automatically a value of 0.10
    :param ref_freq:            If > 0 will be generated at least a reference frame every "ref_freq" frames.
                                range [0-1500]. If is not set, is assigned automatically a value depending on ref_merge
                                value and method: if ex_model in (0, 2) 10 else 0.
    :param ref_norm:            If true the B&W frames are normalized before apply the Scene Detection to generate the
                                reference frames. The normalization will increase the sensitivity to smooth scene changes,
                                range [True, False], default: False
    :param max_memory_frames:   Parameter used by ColorMNet/DeepRemaster models.
                                For ColorMNet specify the max number of encoded frames to keep in memory.
                                Its value depend on encode mode and must be defined manually following the suggested values.
                                encode_mode=0: there is no memory limit (it could be all the frames in the clip).
                                Suggested values are:
                                    min=150, max=10000
                                If = 0 will be filled with the value of 1000 or the clip length if lower.
                                encode_mode=1: the max memory frames is limited by available GPU memory.
                                Suggested values are:
                                    min=1, max=4    : for 8GB GPU
                                    min=1, max=8    : for 12GB GPU
                                    min=1, max=15   : for 24GB GPU
                                If = 0 will be filled with the max value (depending on total GPU RAM available).
                                For DeepRemaster represent the number to reference frames to keep in memory.
                                Suggested values are:
                                    min=1, max=100
    :param render_vivid:        Depending on selected ex_model, if enabled (True):
                                    0) CMNET2: the saturation will be increased by about 15%.
                                    1) Deep-Exemplar: the saturation will be increased by about 25%.
                                    2) Deep-Remaster: the saturation will be increased by about 20% and Hue by +10.
                                    3) ColorMNet: the frames memory is reset at every reference frame update
                                range [True, False]. Default = False
    :param encode_mode:         Parameter used by CMNET2/ColorMNet to define the encode mode strategy.
                                Available values are:
                                     0: remote encoding. The frames will be colored by a thread outside Vapoursynth.
                                                         This option don't have any GPU memory limitation and will allow
                                                         to fully use the long term frame memory.
                                                         It is the faster encode method (default)
                                     1: local encoding.  The frames will be colored inside the Vapoursynth environment.
                                                         In this case the max_memory will be limited by the size of GPU
                                                         memory (max 15 frames for 24GB GPU).
                                                         Useful for coloring clips with a lot of smooth transitions,
                                                         since in this case is better to use a short frame memory or
                                                         the Deep-Exemplar model, which is faster.
    :param retry_threshold:     (CMNET2 only) Threshold used to identify frames that may benefit from an additional
                                reference frame. Range [0.0, 1.0], default 0.0 (disabled).
                                High values (> 0.3) trigger more retry, while lower values (< 0.3) trigger less retry.
                                Suggested value in the range: 0.20-0.35.
    :param retry_model:         (CMNET2 only) If retry_threshold > 0 it represents the model used to colorize the missing
                                reference frames. Allowed values, are:
                                     0 : Model HAVC (colorization with: 60%*DeOldify + 40%*DDColor)
                                     1 : Model DiT colorization with model_precision = "fp4" (RTX 50-Series)
                                     2 : Model DiT colorization with model_precision = "int4" (RTX 30/40-Series)
                                For the models 1 and 2 is necessary to run the DiT Server as explained in the docstring
                                of HAVC_cmnet2dit(). In the case the DiT Server is not running will be used the model 0
                                (Model HAVC). Range [0, 1, 2], default = 0
    :param torch_dir:           torch hub dir location, default is model directory, if set to None will switch
                                to torch cache dir
    """
    # disable packages warnings
    disable_warnings()

    if method not in (5, 6):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC: Video restore is supported only with methods: 5, 6")

    if not torch.cuda.is_available():
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_deepex: CUDA is not available")

    clip, orig_fmt = convert_format_RGB24(clip)

    clip_ref, orig_fmt_r = convert_format_RGB24(clip_ref)

    if clip_ref.width != clip.width or clip_ref.height != clip.height:
        clip_ref = vs.core.resize.Spline36(clip=clip_ref, width=clip.width, height=clip.height)

    if torch_dir is not None:
        torch.hub.set_dir(torch_dir)

    # static params
    enable_resize = False

    if ref_thresh is None or ref_thresh == 0:
        ref_thresh = constants.DEF_THRESHOLD
    if ref_freq is None or ref_freq == 0:
        if ex_model in (0, 2):
            ref_freq = constants.DEF_MIN_FREQ
        else:
            ref_freq = 0

    refmerge_weight: list[float] = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]
    if ref_merge == 0 or method in (2, 6):
        clip_ref = SceneDetect(clip_ref, threshold=ref_thresh, frequency=ref_freq, frame_norm=ref_norm)
        ref_weight = 1.0
        clip_sc = None
    else:
        if ref_weight is None or ref_weight == 0:
            ref_weight = refmerge_weight[ref_merge]
        clip_ref = SceneDetect(clip_ref, threshold=0, frequency=1)
        clip_sc = SceneDetect(clip_ref, threshold=ref_thresh, frequency=ref_freq, frame_norm=ref_norm)

    clip = CopySCDetect(clip, clip_ref)

    clip_orig = clip
    if ex_model == 0:
        d_size = get_render_size(clip.width, clip.height, render_speed=render_speed.lower())
        clip = clip.resize.Spline36(width=d_size[0], height=d_size[1])
        clip_ref = clip_ref.resize.Spline36(width=d_size[0], height=d_size[1])
    else:
        d_size = get_deepex_size(render_speed=render_speed.lower(), enable_resize=enable_resize, ex_model=ex_model)
        smc = SmartResizeColorizer(clip_size=d_size, ex_model=ex_model)
        smr = SmartResizeReference(clip_size=d_size, ex_model=ex_model)
        # clip and clip_ref are resized to match the frame size used for inference
        clip = smc.get_resized_clip(clip)
        clip_ref = smr.get_resized_clip(clip_ref)

    ref_same_as_video = False

    match ex_model:
        case 0:  # CMNET2
            clip_colored = vs_colormnet2(clip, clip_ref, clip_sc, image_size=-1, enable_resize=enable_resize,
                                         encode_mode=encode_mode, max_memory_frames=max_memory_frames,
                                         frame_propagate=ref_same_as_video, render_vivid=render_vivid,
                                         ref_weight=ref_weight,  retry_perm_share_threshold=retry_threshold, retry_model=retry_model)
        case 1:  # Deep-Exemplar
            clip_colored = vs_deepex(clip, clip_ref, clip_sc, image_size=d_size, enable_resize=enable_resize,
                                     propagate=ref_same_as_video, wls_filter_on=True, render_vivid=render_vivid,
                                     ref_weight=ref_weight)
        case 2:  # DeepRemaster
            clip_colored = vs_deepremaster(clip, clip_ref, clip_sc, render_vivid=render_vivid, ref_weight=ref_weight,
                                           memory_size=max_memory_frames, ref_frequency=ref_freq)
        case 3:  # ColorMNet
            clip_colored = vs_colormnet(clip, clip_ref, clip_sc, image_size=-1, enable_resize=enable_resize,
                                            encode_mode=encode_mode, max_memory_frames=max_memory_frames,
                                            frame_propagate=ref_same_as_video, render_vivid=render_vivid,
                                            ref_weight=ref_weight)
        case _:
            clip_colored = None
            HAVC_LogMessage(MessageType.EXCEPTION, "HybridAVC: unknown exemplar model id: " + str(ex_model))

    if ex_model == 0:
        clip_resized = clip_colored.resize.Spline36(width=clip_orig.width, height=clip_orig.height)
    else:
        clip_resized = smc.restore_clip_size(clip_colored)

    # restore original resolution details, 5% faster than ShufflePlanes()
    clip_new = vs_recover_clip_luma(clip_orig, clip_resized)

    return restore_format(clip_new, orig_fmt)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
coloring function with additional pre-process and post-process filters 
"""

def vs_frame_interpolation(clip: vs.VideoNode, clip_ref: vs.VideoNode, frame_interp: int = 5,
                           chroma_adjust: str = "none", process_id: int = 1)  -> vs.VideoNode:
    """Internal helper that selects the appropriate frame-interpolation backend.

    Routes to Deep-Exemplar (ex_model=1) when frame_interp < 5, or to CMNET2/ColorMNet
    when frame_interp >= 5. The choice between CMNET2 and ColorMNet for the latter case
    is controlled by process_id (1 = CMNET2, 2 = ColorMNet via HAVC_cmnet1).

    :param clip:          B&W source clip (RGB24).
    :param clip_ref:      Coloured reference clip with scene-change frame props.
    :param frame_interp:  Interpolation frequency in frames [1, 10]. Values < 5 use
                          Deep-Exemplar; values >= 5 use CMNET2/ColorMNet.
    :param chroma_adjust: Chroma adjustment string applied as colormap. 'none' = disabled.
    :param process_id:    Backend selector when frame_interp >= 5: 1 = CMNET2, 2 = ColorMNet.
    :return:              Colour-interpolated RGB24 clip.
    """
    if frame_interp < 5:  # Deep-Exemplar
        clip_colored = HAVC_deepex(clip=clip, clip_ref=clip_ref, method=0, render_speed='Medium', render_vivid=True,
                                       ref_merge=0, sc_framedir=None, only_ref_frames=False, dark=False,
                                       ref_thresh=0.10, ex_model=1, encode_mode=0, max_memory_frames=0,
                                       ref_freq=frame_interp, ref_norm=False, smooth=False, colormap=chroma_adjust)
    else:
        if process_id == 1:
            clip_colored = HAVC_deepex(clip=clip, clip_ref=clip_ref, method=0, render_speed='Medium', render_vivid=False,
                                       ref_merge=0, sc_framedir=None, only_ref_frames=False, dark=False,
                                       ref_thresh=0.10, ex_model=0, encode_mode=0, max_memory_frames=0,
                                       ref_freq=frame_interp*2, ref_norm=False, smooth=False, colormap=chroma_adjust)
        else:
            clip_colored = HAVC_cmnet1(clip=clip, clip_ref=clip_ref, render_speed='Medium', render_vivid=True,
                                   ref_merge=0, dark=True, dark_p=[0.2, 0.8], ref_thresh=0.10,
                                   encode_mode=0, max_memory_frames=0, ref_freq=frame_interp*2, ref_norm=True,
                                   smooth=True, smooth_p=[0.3, 0.7, 0.9, 0.0, "none"], colormap=chroma_adjust)

    return clip_colored


def HAVC_colorizer_fast(
        clip: vs.VideoNode, method: int = 2, mweight: float = 0.4, deoldify_p: list = (0, 24, 1.0, 0.0),
        ddcolor_p: list = (1, 24, 1.0, 0.0, True), ddtweak: list[bool] = (False, False, False),
        ddtweak_p: list = (constants.DEF_TWEAK_p, "300:360|0.8,0.1"), frame_interp: int = 5, chroma_adjust: str = "none",
        debug_level: int = 0) -> vs.VideoNode:
    """A Deep Learning based project for colorizing and restoring old images and video using Deoldify and DDColor

    :param clip:                clip to process, any format is supported
    :param method:              method used to combine deoldify() with ddcolor() (default = 2):
                                    0 : deoldify only (no merge)
                                    1 : ddcolor only (no merge)
                                    2 : Simple Merge (default):
                                        the frames are combined using a weighted merge, where the parameter "mweight"
                                        represent the weight assigned to the colors provided by the ddcolor() frames.
                                        With this method is suggested a starting weight < 50% (ex. = 40%).
                                    3 : Constrained Chroma Merge:
                                        given that the colors provided by deoldify() are more conservative and stable
                                        than the colors obtained with ddcolor(). The frames are combined by assigning
                                        a limit to the amount of difference in chroma values between deoldify() and
                                        ddcolor() this limit is defined by the threshold parameter "cmc_tresh".
                                        The limit is applied to the image converted to "YUV". For example when
                                        cmc_tresh=0.2, the chroma values "U","V" of ddcolor() frame will be constrained
                                        to have an absolute percentage difference respect to "U","V" provided by deoldify()
                                        not higher than 20%. The final limited frame will be merged again with the deoldify()
                                        frame. With this method is suggested a starting weight > 50% (ex. = 60%).
                                    4 : Luma Masked Merge:
                                        the frames are combined using a masked merge, the pixels of ddcolor() with luma < "luma_mask_limit"
                                        will be filled with the pixels of deoldify(). If "luma_white_limit" > "luma_mask_limit" the mask will
                                        apply a gradient till "luma_white_limit". If the parameter "mweight" > 0 the final masked frame will
                                        be merged again with the deoldify() frame. With this method is suggested a starting weight > 60%
                                        (ex. = 70%).
                                    5 : Adaptive Luma Merge:
                                        given that the ddcolor() perfomance is quite bad on dark scenes, the images are
                                        combined by decreasing the weight assigned to ddcolor() when the luma is
                                        below a given threshold given by: luma_threshold. The weight is calculated using
                                        the formula: merge_weight = max(mweight * (luma/luma_threshold)^alpha, min_weight).
                                        For example with: luma_threshold = 0.6 and alpha = 1, the weight assigned to
                                        ddcolor() will start to decrease linearly when the luma < 60% till "min_weight".
                                        For alpha=2, begins to decrease quadratically (because luma/luma_threshold < 1).
                                        With this method is suggested a starting weight > 70% (ex. = 80%).
                                    6 : Chroma Retention Merge:
                                        Given that the colors provided by deoldify() are more conservative and stable
                                        than the colors obtained with ddcolor(). This function try to restore the
                                        colors of gray pixels provide by deoldify() by using the colors provided
                                        by ddcolor(). The gray pixels are identified by the parameter "tht". Once are
                                        identified the gray pixels are substituted with the desaturated colors in deoldify(),
                                        the level of desaturation is identified by the parameter "sat". It is performed
                                        a "gradient" substitution, i.e. the gray pixels are gradually substituted depending
                                        on the level of gray gradient. The steepness of gradient curve is controlled by
                                        the parameter "alpha". Optionally is possible to resize the frame before the filter
                                        application to speed up the filter by setting True the parameter "chroma_resize".
                                     7 : ChromaBound Adaptive
                                        Adaptive version of Constrained-Chroma. In this version the chroma tolerance is
                                        adaptive, i.e., it is applied an approach that will allow more color variation
                                        in textured/complex regions and less in smooth areas. The texture strength is
                                        computed via Laplacian and chroma tolerance is controlled by the following
                                        parameters:
                                              [2] base_tol: int = 20,  # Base chroma tolerance (smooth areas)
                                              [3] max_extra: int = 24,  # Extra tolerance for textured areas
                                    The methods 3, 4 and 7 are similar to Simple Merge, but before the merge with clipa
                                    the clipb frame is limited in the chroma changes (method 3, 7) or limited based
                                    on theluma (method 4). The method 5 is a Simple Merge where the weight decrease
                                    with luma.
    :param mweight:             weight given to ddcolor's clip in all merge methods, range [0-1] (0.01=1%)
    :param deoldify_p:          parameters for deoldify():
                                   [0] deoldify model to use (default = 0):
                                       0 = ColorizeVideo_gen
                                       1 = ColorizeStable_gen
                                       2 = ColorizeArtistic_gen
                                   [1] render factor for the model, range: 10-44 (default = 24).
                                   [2] saturation parameter to apply to deoldify color model (default = 1)
                                   [3] hue parameter to apply to deoldify color model (default = 0)
    :param ddcolor_p:           parameters for ddcolor():
                                   [0] ddcolor model (default = 1):
                                       0 = ddcolor_modelscope,
                                       1 = ddcolor_artistic
                                       2 = colorization_siggraph17
                                       3 = colorization_eccv16
                                   [1] render factor for the model, if=0 will be auto selected
                                       (default = 24), range: [0, 10-64]
                                   [2] saturation parameter to apply to ddcolor color model (default = 1)
                                   [3] hue parameter to apply to ddcolor color model (default = 0)
                                   [4] enable/disable FP16 in ddcolor inference
    :param ddtweak:             enabled/disable tweak parameters for ddcolor(), range [True,False]
    :param ddtweak_p:           tweak parameters for ddcolor():
                                   [0] : ddcolor tweak's bright (default = 0)
                                   [1] : ddcolor tweak's contrast (default = 1), if < 1 ddcolor provides de-saturated frames
                                   [2] : ddcolor tweak's gamma (default = 1)
                                   [3] : luma constrained gamma -> luma constrained gamma correction enabled (default = False), range: [True, False]
                                            When enabled the average luma of a video clip will be forced to don't be below the value
                                            defined by the parameter "luma_min". The function allow to modify the gamma
                                            of the clip if the average luma is below the parameter "gamma_luma_min". A gamma value > 2.0 improves
                                            the ddcolor stability on bright scenes, while a gamma < 1 improves the ddcolor stability on
                                            dark scenes. The decrease of the gamma with luma is activated using a gamma_alpha != 0.
                                   [4] : luma_min: luma (%) min value for tweak activation (default = 0.2), if=0 is not activated, range [0-1]
                                   [5] : gamma_luma_min: luma (%) min value for gamma tweak activation (default = 0.5), if=0 is not activated, range [0-1]
                                   [6] : gamma_alpha: the gamma will decrease with the luma g = max(gamma * pow(luma/gamma_luma_min, gamma_alpha), gamma_min),
                                         for a movie with a lot of dark scenes is suggested alpha > 1, if=0 is not activated, range [>=0]
                                   [7] : gamma_min: minimum value for gamma, range (default=0.5) [>0.1]
                                   [8] : "chroma adjustment" parameter (optional), if="none" is disabled (see the README)
    :param frame_interp:        This parameter will allow to enable the frame interpolation. This method will use
                                Deep-Exemplar to interpolate the colored frames if frame_interp < 5, otherwise will use
                                ColorMNet. If = 0, the interpolation is disabled, if > 0 represent the number of frames
                                used for interpolation. The quality of interpolation will decrease with the number of
                                frames, suggested value is 3. Range [0-10], Default = 0
    :param chroma_adjust:       Direct hue/color mapping (only on ref-frames), without luma filtering, using the "chroma adjustment"
                                parameter, if="none" is disabled.
    :param debug_level:         Set HAVC debug message level.
    """
    # disable packages warnings
    disable_warnings()

    HAVC_set_debug_level(debug_level)

    if frame_interp not in range(11) or frame_interp == 0:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_colorizer_fast: frame_interp must be in range [1-10]")

    clip, orig_fmt = convert_format_RGB24(clip)

    clip_ref = HAVC_colorizer(clip, method=method, mweight=mweight, deoldify_p=deoldify_p, ddcolor_p=ddcolor_p,
                          ddtweak=ddtweak, ddtweak_p=ddtweak_p, sc_threshold=0.1, sc_tht_offset=1,
                          sc_min_freq=frame_interp, sc_min_int=1, sc_tht_ssim=0.0, sc_normalize=False,
                          debug_level=debug_level)

    clip_colored = vs_frame_interpolation(clip, clip_ref, frame_interp, chroma_adjust, process_id=2)
    clip_colored = clip_colored.std.SetFrameProp(prop="sc_threshold", floatval=0.1)
    clip_colored = clip_colored.std.SetFrameProp(prop="sc_frequency", intval=1)

    return restore_format(clip_colored, orig_fmt)

def HAVC_colorizer(
        clip: vs.VideoNode, method: int = 2, mweight: float = 0.4, deoldify_p: list = (0, 24, 1.0, 0.0),
        ddcolor_p: list = (1, 24, 1.0, 0.0, True), ddtweak: list[bool] = (False, False, False),
        ddtweak_p: list = (constants.DEF_TWEAK_p, "300:360|0.8,0.1"),
        cmc_p: list = constants.DEF_CMC_p, lmm_p: list = constants.DEF_LMM_p, alm_p: list = constants.DEF_ALM_p,
        crt_p: list = constants.DEF_CRT_p, cmb_sw: bool = False, sc_threshold: float = 0.0, sc_tht_offset: int = 1,
        sc_min_freq: int = 0, sc_tht_ssim: float = 0.0, sc_normalize: bool = False, sc_min_int: int = 1,
        sc_tht_white: float = constants.DEF_THT_WHITE, sc_tht_black: float = constants.DEF_THT_BLACK, device_index: int = 0,
        torch_dir: str = model_dir, debug_level: int = 0) -> vs.VideoNode:
    """A Deep Learning based project for colorizing and restoring old images and video using Deoldify and DDColor

    :param clip:                clip to process, any format is supported
    :param method:              method used to combine deoldify() with ddcolor() (default = 2):
                                    0 : deoldify only (no merge)
                                    1 : ddcolor only (no merge)
                                    2 : Simple Merge (default):
                                        the frames are combined using a weighted merge, where the parameter "mweight"
                                        represent the weight assigned to the colors provided by the ddcolor() frames.
                                        With this method is suggested a starting weight < 50% (ex. = 40%).
                                    3 : Constrained Chroma Merge:
                                        given that the colors provided by deoldify() are more conservative and stable
                                        than the colors obtained with ddcolor(). The frames are combined by assigning
                                        a limit to the amount of difference in chroma values between deoldify() and
                                        ddcolor() this limit is defined by the threshold parameter "cmc_tresh".
                                        The limit is applied to the image converted to "YUV". For example when
                                        cmc_tresh=0.2, the chroma values "U","V" of ddcolor() frame will be constrained
                                        to have an absolute percentage difference respect to "U","V" provided by deoldify()
                                        not higher than 20%. The final limited frame will be merged again with the deoldify()
                                        frame. With this method is suggested a starting weight > 50% (ex. = 60%).
                                    4 : Luma Masked Merge:
                                        the frames are combined using a masked merge, the pixels of ddcolor() with luma < "luma_mask_limit"
                                        will be filled with the pixels of deoldify(). If "luma_white_limit" > "luma_mask_limit" the mask will
                                        apply a gradient till "luma_white_limit". If the parameter "mweight" > 0 the final masked frame will
                                        be merged again with the deoldify() frame. With this method is suggested a starting weight > 60%
                                        (ex. = 70%).
                                    5 : Adaptive Luma Merge:
                                        given that the ddcolor() perfomance is quite bad on dark scenes, the images are
                                        combined by decreasing the weight assigned to ddcolor() when the luma is
                                        below a given threshold given by: luma_threshold. The weight is calculated using
                                        the formula: merge_weight = max(mweight * (luma/luma_threshold)^alpha, min_weight).
                                        For example with: luma_threshold = 0.6 and alpha = 1, the weight assigned to
                                        ddcolor() will start to decrease linearly when the luma < 60% till "min_weight".
                                        For alpha=2, begins to decrease quadratically (because luma/luma_threshold < 1).
                                        With this method is suggested a starting weight > 70% (ex. = 80%).
                                    6 : Chroma Retention Merge:
                                        Given that the colors provided by deoldify() are more conservative and stable
                                        than the colors obtained with ddcolor(). This function try to restore the
                                        colors of gray pixels provide by deoldify() by using the colors provided
                                        by ddcolor(). The gray pixels are identified by the parameter "tht". Once are
                                        identified the gray pixels are substituted with the desaturated colors in deoldify(),
                                        the level of desaturation is identified by the parameter "sat". It is performed
                                        a "gradient" substitution, i.e. the gray pixels are gradually substituted depending
                                        on the level of gray gradient. The steepness of gradient curve is controlled by
                                        the parameter "alpha". Optionally is possible to resize the frame before the filter
                                        application to speed up the filter by setting True the parameter chroma_resize.
                                     7 : ChromaBound Adaptive
                                        Adaptive version of Constrained-Chroma. In this version the chroma tolerance is
                                        adaptive, i.e., it is applied an approach that will allow more color variation
                                        in textured/complex regions and less in smooth areas. The texture strength is
                                        computed via Laplacian and chroma tolerance is controlled by the following
                                        parameters:
                                              [2] base_tol: int = 20,  # Base chroma tolerance (smooth areas)
                                              [3] max_extra: int = 24,  # Extra tolerance for textured areas
                                    The methods 3, 4 and 7 are similar to Simple Merge, but before the merge with clipa
                                    the clipb frame is limited in the chroma changes (method 3, 7) or limited based
                                    on theluma (method 4). The method 5 is a Simple Merge where the weight decrease
                                    with luma.
    :param mweight:             weight given to ddcolor's clip in all merge methods, range [0-1] (0.01=1%)
    :param deoldify_p:          parameters for deoldify():
                                   [0] deoldify model to use (default = 0):
                                       0 = ColorizeVideo_gen
                                       1 = ColorizeStable_gen
                                       2 = ColorizeArtistic_gen
                                   [1] render factor for the model, range: 10-44 (default = 24).
                                   [2] saturation parameter to apply to deoldify color model (default = 1)
                                   [3] hue parameter to apply to deoldify color model (default = 0)
    :param ddcolor_p:           parameters for ddcolor():
                                   [0] ddcolor model (default = 1):
                                       0 = ddcolor_modelscope,
                                       1 = ddcolor_artistic
                                       2 = colorization_siggraph17
                                       3 = colorization_eccv16
                                   [1] render factor for the model, if=0 will be auto selected
                                       (default = 24), range: [0, 10-64]
                                   [2] saturation parameter to apply to ddcolor color model (default = 1)
                                   [3] hue parameter to apply to ddcolor color model (default = 0)
                                   [4] enable/disable FP16 in ddcolor inference
    :param ddtweak:             enabled/disable tweak parameters for ddcolor(), range [True,False]
    :param ddtweak_p:           tweak parameters for ddcolor():
                                   [0] : ddcolor tweak's bright (default = 0)
                                   [1] : ddcolor tweak's contrast (default = 1), if < 1 ddcolor provides de-saturated frames
                                   [2] : ddcolor tweak's gamma (default = 1)
                                   [3] : luma constrained gamma -> luma constrained gamma correction enabled (default = False), range: [True, False]
                                            When enabled the average luma of a video clip will be forced to don't be below the value
                                            defined by the parameter "luma_min". The function allow to modify the gamma
                                            of the clip if the average luma is below the parameter "gamma_luma_min". A gamma value > 2.0 improves
                                            the ddcolor stability on bright scenes, while a gamma < 1 improves the ddcolor stability on
                                            dark scenes. The decrease of the gamma with luma is activated using a gamma_alpha != 0.
                                   [4] : luma_min: luma (%) min value for tweak activation (default = 0.2), if=0 is not activated, range [0-1]
                                   [5] : gamma_luma_min: luma (%) min value for gamma tweak activation (default = 0.5), if=0 is not activated, range [0-1]
                                   [6] : gamma_alpha: the gamma will decrease with the luma g = max(gamma * pow(luma/gamma_luma_min, gamma_alpha), gamma_min),
                                         for a movie with a lot of dark scenes is suggested alpha > 1, if=0 is not activated, range [>=0]
                                   [7] : gamma_min: minimum value for gamma, range (default=0.5) [>0.1]
                                   [8] : "chroma adjustment" parameter (optional), if="none" is disabled (see the README)
    :param cmc_p:               parameters list for method: "Constrained Chroma Merge", "ChromaBound Adaptive"
                                (see methods 3, 7 for a full explanation):
                                      [0] chroma_threshold (%), range [0-1] (0.01=1%), default = 0.15
                                      [1] red_fix (default = True),  # if true red-regions in dark areas are desaturated
                                      [2] base_tol (default = 20),  # Base chroma tolerance (smooth areas)
                                      [3] max_extra: (default = 24),  # Extra tolerance for textured areas
    :param lmm_p:               parameters for method: "Luma Masked Merge" (see method=4 for a full explanation)
                                   [0] : luma_mask_limit: luma limit for build the mask used in Luma Masked Merge, range [0-1] (0.01=1%)
                                   [1] : luma_white_limit: the mask will apply a gradient till luma_white_limit, range [0-1] (0.01=1%)
                                   [2] : luma_mask_sat: if < 1 the ddcolor dark pixels will substitute with the desaturated deoldify pixels, range [0-1] (0.01=1%)
    :param alm_p:               parameters for method: "Adaptive Luma Merge" (see method=5 for a full explanation)
                                   [0] : luma_threshold: threshold for the gradient merge, range [0-1] (0.01=1%)
                                   [1] : alpha: exponent parameter used for the weight calculation, range [>0]
                                   [2] : min_weight: min merge weight, range [0-1] (0.01=1%)
    :param crt_p:               parameters for method: "Chroma Retention Merge" (see method=6 for a full explanation)
                                   [0] : sat: this parameter allows to change the saturation of colored clip (default = 0.8)
                                   [1] : tht: threshold to identify gray pixels, range[0, 255] (default = 30)
                                   [2] : alpha: parameter used to control the steepness of gradient curve, range [>0] (default = 2.0)
                                   [3] : chroma_resize: if True, the frames will be resized to improve the filter speed (default = False)
    :param cmb_sw:              if true switch the clip order in all the combining methods, range [True,False]
    :param sc_threshold:        Scene change threshold used to generate the reference frames to be used by
                                "Exemplar-based" Video Colorization. It is a percentage of the luma change between
                                the previous and the current frame. range [0-1], default 0.0. If =0 are not generate
                                reference frames and will be colorized all the frames.
    :param sc_tht_offset:       Offset index used for the Scene change detection. The comparison will be performed,
                                between frame[n] and frame[n-offset]. An offset > 1 is useful to detect blended scene
                                change, range[1, 25]. Default = 1.
    :param sc_tht_ssim:         Threshold used by the SSIM (Structural Similarity Index Metric) selection filter.
                                If > 0, will be activated a filter that will improve the scene-change detection,
                                by discarding images that are similar.
                                Suggested values are between 0.35 and 0.85, range [0-1], default 0.0 (deactivated)
    :param sc_normalize:        If true the B&W frames are normalized before use misc.SCDetect(), the normalization will
                                increase the sensitivity to smooth scene changes.
    :param sc_min_int:          Minimum number of frame interval between scene changes, range[1, 25]. Default = 1.
    :param sc_min_freq:         if > 0 will be generate at least a reference frame every "sc_min_freq" frames.
                                range [0-1500], default: 0.
    :param sc_tht_white:        Threshold to identify white frames, range [0-1], default 0.85.
    :param sc_tht_black:        Threshold to identify dark frames, range [0-1], default 0.15.
    :param device_index:        device ordinal of the GPU, choices: GPU0...GPU7, CPU=99 (default = 0)
    :param torch_dir:           torch hub dir location, default is model directory, if set to None will switch to torch cache dir.
    :param debug_level:         Set the level of HAVC debug messages. Default = 0 (no messages).
    """
    # disable packages warnings
    disable_warnings()

    HAVC_set_debug_level(debug_level)

    if not torch.cuda.is_available() and device_index != 99:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_colorizer: CUDA is not available")

    clip, orig_fmt = convert_format_RGB24(clip)

    if sc_threshold < 0:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_colorizer: sc_threshold must be >= 0")

    if sc_min_freq < 0:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_colorizer: sc_min_freq must be >= 0")

    if method == 0:
        merge_weight = 0.0
    elif method == 1:
        merge_weight = 1.0
    else:
        merge_weight = mweight

    if merge_weight == 0.0:
        method = 0  # deoldify only
    elif merge_weight == 1.0:
        method = 1  # ddcolor only

    # unpack deoldify_params
    deoldify_model = deoldify_p[0]
    deoldify_rf = deoldify_p[1]
    deoldify_sat = deoldify_p[2]
    deoldify_hue = deoldify_p[3]

    # unpack deoldify_params
    ddcolor_model = ddcolor_p[0]
    ddcolor_rf = ddcolor_p[1]
    ddcolor_sat = ddcolor_p[2]
    ddcolor_hue = ddcolor_p[3]
    ddcolor_enable_fp16 = ddcolor_p[4]

    if os.path.getsize(os.path.join(model_dir, "ColorizeVideo_gen.pth")) == 0:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_colorizer: model files have not been downloaded.")

    if device_index > 7 and device_index != 99:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_colorizer: wrong device_index, choices are: GPU0...GPU7, CPU=99")

    if ddcolor_rf != 0 and ddcolor_rf not in range(10, 65):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_colorizer: ddcolor render_factor must be between: 10-64")

    # choices: GPU0...GPU7, CPU=99
    device.set(device=DeviceId(device_index))

    if torch_dir is not None:
        torch.hub.set_dir(torch_dir)

    if ddcolor_rf == 0:
        ddcolor_rf = min(max(math.trunc(0.4 * clip.width / 16), 16), 32)

    scenechange = not (sc_threshold == 0 and sc_min_freq == 0)

    if scenechange:
        clip = SceneDetect(clip, threshold=sc_threshold, frequency=sc_min_freq, sc_tht_filter=sc_tht_ssim,
                           tht_offset=sc_tht_offset, min_length=sc_min_int, frame_norm=sc_normalize,
                           tht_white=sc_tht_white, tht_black=sc_tht_black, sc_debug=(debug_level==constants.DEF_LEVEL_DEBUG))

    frame_size = min(max(ddcolor_rf, deoldify_rf) * 16, clip.width)  # frame size calculation for inference()
    clip_orig = clip
    clip = clip.resize.Spline64(width=frame_size, height=frame_size)

    clipa = vs_sc_deoldify(clip, method=method, model=deoldify_model, render_factor=deoldify_rf,
                           scenechange=scenechange, package_dir=package_dir)
    clipb = vs_sc_ddcolor(clip, method=method, model=ddcolor_model, render_factor=ddcolor_rf, tweaks_flags=ddtweak,
                          tweaks=ddtweak_p, enable_fp16=ddcolor_enable_fp16, scenechange=scenechange,
                          device_index=device_index)

    if scenechange:
        clip_colored = vs_sc_combine_models(clip_a=clipa, clip_b=clipb, method=method, sat=[deoldify_sat, ddcolor_sat],
                                         hue=[deoldify_hue, ddcolor_hue], clipb_weight=merge_weight, CMC_p=cmc_p,
                                         LMM_p=lmm_p, ALM_p=alm_p, CRT_p=crt_p, invert_clips=cmb_sw, scenechange=True)
    else:
        clip_colored = vs_combine_models(clip_a=clipa, clip_b=clipb, method=method, sat=[deoldify_sat, ddcolor_sat],
                                         hue=[deoldify_hue, ddcolor_hue], clipb_weight=merge_weight, CMC_p=cmc_p,
                                         LMM_p=lmm_p, ALM_p=alm_p, CRT_p=crt_p, invert_clips=cmb_sw)

    clip_resized = _clip_chroma_resize(clip_orig, clip_colored)

    return restore_format(clip_resized, orig_fmt)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function with HAVC merge methods  
"""


def HAVC_merge(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, clip_luma: vs.VideoNode = None, weight: float = 0.5,
               method: int = 2, cmc_p: list = constants.DEF_CMC_p, lmm_p: list = constants.DEF_LMM_p, alm_p: list = constants.DEF_ALM_p,
               crt_p: list = constants.DEF_CRT_p) -> vs.VideoNode:
    """Utility function with the implementation of HAVC merge methods

    :param clipa:               first clip to merge, any format is supported
    :param clipb:               second clip to merge, any format is supported
    :param clip_luma:           if specified, clip_luma will be used as source of luma component for the merge. It is an
                                optional parameter, and it is suggested to provide the clip with the best luma
                                resolution between clipa and clipb. It is used only with the methods: 3, 4, 5 and can
                                speed up the filter when it uses these methods.
    :param method:              method used to combine clipa with clipb (default = 2):
                                    0 : clipa only (no merge), if clip_luma is not None is luma merged with clip_luma
                                    1 : clipb only (no merge), if clip_luma is not None is luma merged with clip_luma
                                    2 : Simple Merge (default):
                                        the frames are combined using a weighted merge, where the parameter "weight"
                                        represent the weight assigned to the colors provided by the clipb frames.
                                        If weight = 0 will be returned clipa, if = 1 will be returned clipb.
                                    3 : Constrained Chroma Merge:
                                        The frames are combined by assigning a limit to the amount of difference in
                                        chroma values between clipa and clipb this limit is defined by the threshold
                                        parameter "cmc_tresh".
                                        The limit is applied to the image converted to "YUV". For example when
                                        cmc_tresh=0.2, the chroma values "U","V" of clipb frame will be constrained
                                        to have an absolute percentage difference respect to "U","V" provided by clipa
                                        not higher than 20%. The final limited frame will be merged again with the clipa
                                        frame. With this method is suggested a starting weight > 50% (ex. = 60%).
                                    4 : Luma Masked Merge:
                                        the frames are combined using a masked merge, the pixels of clipb with
                                        luma < "luma_mask_limit" will be filled with the pixels of clipa.
                                        If "luma_white_limit" > "luma_mask_limit" the mask will apply a gradient till
                                        "luma_white_limit". If the parameter "weight" > 0 the final masked frame will
                                        be merged again with the clipa frame.
                                    5 : Adaptive Luma Merge:
                                        The frames are combined by decreasing the weight assigned to clipb when the
                                        luma is below a given threshold given by: luma_threshold. The weight is
                                        calculated using the formula:
                                            merge_weight = max(weight * (luma/luma_threshold)^alpha, min_weight).
                                        For example with: luma_threshold = 0.6 and alpha = 1, the weight assigned to
                                        clipb will start to decrease linearly when the luma < 60% till "min_weight".
                                        For alpha=2, begins to decrease quadratically (because luma/luma_threshold < 1).
                                    6 : Chroma Retention Merge:
                                        Given that the colors provided by deoldify() are more conservative and stable
                                        than the colors obtained with ddcolor(). This function try to restore the
                                        colors of gray pixels provide by deoldify() by using the colors provided
                                        by ddcolor(). The gray pixels are identified by the parameter "tht". Once are
                                        identified the gray pixels are substituted with the desaturated colors in deoldify(),
                                        the level of desaturation is identified by the parameter "sat". It is performed
                                        a "gradient" substitution, i.e. the gray pixels are gradually substituted depending
                                        on the level of gray gradient. The steepness of gradient curve is controlled by
                                        the parameter "alpha". Optionally is possible to resize the frame before the filter
                                        application to speed up the filter by setting True the parameter chroma_resize.
                                    7 : ChromaBound Adaptive
                                        Adaptive version of Constrained-Chroma. In this version the chroma tolerance is
                                        adaptive, i.e., it is applied an approach that will allow more color variation
                                        in textured/complex regions and less in smooth areas. The texture strength is
                                        computed via Laplacian and chroma tolerance is controlled by the following
                                        parameters:
                                              [2] base_tol: int = 20,  # Base chroma tolerance (smooth areas)
                                              [3] max_extra: int = 24,  # Extra tolerance for textured areas
                                    The methods 3, 4 and 7 are similar to Simple Merge, but before the merge with clipa
                                    the clipb frame is limited in the chroma changes (method 3, 7) or limited based
                                    on theluma (method 4). The method 5 is a Simple Merge where the weight decrease
                                    with luma.
    :param weight:              weight given to clipb in all merge methods. If weight = 0 will be returned
                                clipa, if = 1 will be returned clipb. range [0-1] (0.01=1%)
    :param cmc_p:               parameters list for method: "Constrained Chroma Merge", "ChromaBound Adaptive"
                                (see methods 3, 7 for a full explanation):
                                      [0] chroma_threshold (%), range [0-1] (0.01=1%), default = 0.15
                                      [1] red_fix (default = True),  # if true red-regions in dark areas are desaturated
                                      [2] base_tol (default = 20),  # Base chroma tolerance (smooth areas)
                                      [3] max_extra: (default = 24),  # Extra tolerance for textured areas
    :param lmm_p:               parameters for method: "Luma Masked Merge" (see method=4 for a full explanation)
                                   [0] : luma_mask_limit: luma limit for build the mask used in Luma Masked Merge,
                                         range [0-1] (0.01=1%)
                                   [1] : luma_white_limit: the mask will apply a gradient till luma_white_limit,
                                         range [0-1] (0.01=1%)
                                   [2] : luma_mask_sat: if < 1 the clipb dark pixels will substitute with the
                                         desaturated clipa pixels, range [0-1] (0.01=1%)
    :param alm_p:               parameters for method: "Adaptive Luma Merge" (see method=5 for a full explanation)
                                   [0] : luma_threshold: threshold for the gradient merge, range [0-1] (0.01=1%)
                                   [1] : alpha: exponent parameter used for the weight calculation, range [>0]
                                   [2] : min_weight: min merge weight, range [0-1] (0.01=1%)
    :param crt_p:               parameters for method: "Chroma Retention Merge" (see method=6 for a full explanation)
                                   [0] : sat: this parameter allows to change the saturation of colored clip (default = 0.8)
                                   [1] : tht: threshold to identify gray pixels, range[0, 255] (default = 30)
                                   [2] : alpha: parameter used to control the steepness of gradient curve, range [>0] (default = 2.0)
                                   [3] : chroma_resize: if True, the frames will be resized to improve the filter speed (default = False)
    """
    # disable packages warnings
    disable_warnings()

    if clipa is not None and not isinstance(clipa, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_merge: this is not a clip: clipa")

    if clipb is not None and not isinstance(clipb, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_merge: this is not a clip: clipb")

    if clip_luma is not None and not isinstance(clip_luma, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_merge: this is not a clip: clip_luma")

    if method == 0 or weight == 0:
        if clip_luma is not None:
            clipa_sc = clipa
            clipa = _clip_chroma_resize(clip_luma, clipa)
            clipa = CopySCDetect(clipa, clipa_sc)
        return clipa

    if method == 1 or weight == 1:
        if clip_luma is not None:
            clipb_sc = clipb
            clipb = _clip_chroma_resize(clip_luma, clipb)
            clipb = CopySCDetect(clipb, clipb_sc)
        return clipb

    merge_weight = weight

    clip_a, orig_fmt_a = convert_format_RGB24(clipa)
    clip_b, orig_fmt_b = convert_format_RGB24(clipb)

    if method == 2:
        clip_merged = vs_simple_merge(clip_a, clip_b, merge_weight)
        return restore_format(clip_merged, orig_fmt_a)

    if clip_luma is not None:
        rf = min(max(math.trunc(0.4 * clip_luma.width / 16), 16), 32)
        frame_size = min(rf * 16, clip_luma.width)
        clip_a = clip_a.resize.Spline64(width=frame_size, height=frame_size)
        clip_b = clip_b.resize.Spline64(width=frame_size, height=frame_size)

    clip_merged = vs_combine_models(clip_a=clip_a, clip_b=clip_b, method=method, sat=[1, 1],
                                    hue=[0, 0], clipb_weight=merge_weight, CMC_p=cmc_p,
                                    LMM_p=lmm_p, ALM_p=alm_p, CRT_p=crt_p, invert_clips=False)

    if clip_luma is not None:
        clipm_sc = clip_merged
        clip_merged = _clip_chroma_resize(clip_luma, clip_merged)
        clip_merged = CopySCDetect(clip_merged, clipm_sc)

    return restore_format(clip_merged, orig_fmt_a)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
Function to perform colorization with DeepRemaster: : Temporal Source-Reference 
Attention Networks for Comprehensive Video Enhancement.
"""


def HAVC_DeepRemaster(
        clip: vs.VideoNode,
        length: int = 2,
        render_vivid: bool = False,
        ref_dir: str = None,
        ref_minedge: int = 256,
        frame_mindim: int = 320,
        ref_buffer_size: int = 20,
        device_index: int = 0,
        inference_mode: bool = False,
        ref_mode: int = 0
) -> vs.VideoNode:
    """Function to perform colorization with DeepRemaster using direct access to RF or using Vapoursynth clips

     :param clip:            Clip to process, any format is supported.
     :param length:          Sequence length that the model processes (min. 2, max. 5). Default: 2
     :param render_vivid:    Given that the generated colors by the inference are a little washed out, by enabling
                             this parameter, the saturation will be increased by about 10%. range [True, False]
     :param ref_dir:         Path of the reference frames, must be of the same size of input clip: Default: None
     :param ref_minedge:     min dimension of reference frames used for inference. Default: 256
     :param frame_mindim:    min dimension of input frames used for inference. Default: 320
     :param ref_buffer_size: reference frame buffer size for inference. Default: 20
     :param device_index:    Device ordinal of the GPU (if = -1 CPU mode is enabled). Default: 0
     :param inference_mode:  Enable/Disable torch inference mode. Default: False
     :param ref_mode:        Mode selected to access to the external reference frames.
                             Allowed values are:
                                 0: will use direct access to reference frame folder (fast)
                                 1: will use Vapoursynth clips to access to reference frames (slow)
     """
    clip, orig_fmt = convert_format_RGB24(clip)

    if ref_dir is None:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_DeepRemaster: ref_dir is unset")

    if ref_mode == 0:
        return remaster.vs_remaster_colorize(clip, length, render_vivid, ref_dir, ref_minedge,
                                               frame_mindim, ref_buffer_size, device_index, inference_mode)

    clip = SceneDetectFromDir(clip, sc_framedir=ref_dir, merge_ref_frame=False, ref_frame_ext=True)

    clip_ref = vs_ext_reference_clip(clip, sc_framedir=ref_dir, clip_resize=True)

    clip_new = remaster.vs_sc_remaster_colorize(clip, clip_ref, clip_sc=None, length=length, render_vivid=render_vivid,
                                                  ref_minedge=ref_minedge, frame_mindim=frame_mindim, ref_buffer_size=ref_buffer_size,
                                                  device_index=0, inference_mode=False)

    return restore_format(clip_new, orig_fmt)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Video color stabilization filter.
"""


def HAVC_stabilizer(clip: vs.VideoNode, dark: bool = False, dark_p: list = (0.2, 0.8), smooth: bool = False,
                    smooth_p: list = (0.3, 0.7, 0.9, 0.0, "none"), stab: bool = False,
                    stab_p: list = (5, 'A', 1, 15, 0.2, 0.8), colormap: str = "none",
                    render_factor: int = 24) -> vs.VideoNode:
    """Video color stabilization filter, which can be applied to stabilize the chroma components in colored clips.
        :param clip:                clip to process, any format is supported.
        :param dark:                enable/disable darkness filter, range [True,False]
        :param dark_p:              parameters for darken the clip's dark portions, which sometimes are wrongly colored by the color models
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0.1-0.5] (0.01=1%)
                                      [1] : dark_amount: amount of desaturation to apply to the dark area, range [0-1]
                                      [2] : "chroma range" parameter (optional), if="none" is disabled (see the README)
        :param smooth:              enable/disable chroma smoothing, range [True, False]
        :param smooth_p:            parameters to adjust the saturation and "vibrancy" of the clip.
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0-1] (0.01=1%)
                                      [1] : white_threshold, if > dark_threshold will be applied a gradient till white_threshold, range [0-1] (0.01=1%)
                                      [2] : dark_sat, amount of de-saturation to apply to the dark area, range [0-1]
                                      [3] : dark_bright, darkness parameter it used to reduce the "V" component in "HSV" colorspace, range [0, 1]
                                      [4] : "chroma range" parameter (optional), if="none" is disabled (see the README)
        :param stab:                enable/disable chroma stabilizer, range [True, False]
        :param stab_p:              parameters for the temporal color stabilizer
                                      [0] : nframes, number of frames to be used in the stabilizer, range[3-15]
                                      [1] : mode, type of average used by the stabilizer: range['A'='arithmetic', 'W'='weighted']
                                      [2] : sat: saturation applied to the restored gray pixels [0,1]
                                      [3] : tht, threshold to detect gray pixels, range [0,255], if=0 is not applied the restore,
                                            its value depends on merge method used, suggested values are:
                                                method 0: tht = 5
                                                method 1: tht = 60 (ddcolor provides very saturated frames)
                                                method 2: tht = 15
                                                method 3: tht = 20
                                                method 4: tht = 5
                                                method 5: tht = 10
                                      [4] : weight, weight to blend the restored imaage (default=0.2), range [0-1], if=0 is not applied the blending
                                      [5] : tht_scen, threshold for scene change detection (default = 0.8), if=0 is not activated, range [0.01-0.50]
                                      [6] : "chroma adjustment" parameter (optional), if="none" is disabled (see the README)
        :param colormap:            direct hue/color mapping, without luma filtering, using the "chroma adjustment" parameter, if="none" is disabled
        :param render_factor:       render_factor to apply to the filters, the frame size will be reduced to speed-up the filters,
                                    but the final resolution will be the one of the original clip. If = 0 will be auto selected.
                                    This approach takes advantage of the fact that human eyes are much less sensitive to
                                    imperfections in chrominance compared to luminance. This means that it is possible to speed-up
                                    the chroma filters and and ultimately get a great high-resolution result, range: [0, 10-64]
    """

    clip, orig_fmt = convert_format_RGB24(clip)

    # enable chroma_resize
    chroma_resize_enabled = True

    if render_factor != 0 and render_factor not in range(16, 65):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_stabilizer: render_factor must be between: 16-64")

    if render_factor == 0:
        render_factor = min(max(math.trunc(0.4 * clip.width / 16), 16), 32)

    clip_orig = clip
    if chroma_resize_enabled:
        frame_size = min(render_factor * 16, clip.width)  # frame size calculation for filters
        clip = clip.resize.Spline64(width=frame_size, height=frame_size)

    # unpack dark
    dark_enabled = dark
    dark_threshold = dark_p[0]
    dark_amount = dark_p[1]
    if len(dark_p) > 2:
        dark_hue_adjust = dark_p[2]
    else:
        dark_hue_adjust = 'none'

    # unpack chroma_smoothing
    chroma_smoothing_enabled = smooth
    black_threshold = smooth_p[0]
    white_threshold = smooth_p[1]
    dark_sat = smooth_p[2]
    dark_bright = -smooth_p[3]  # change the sign to reduce the bright
    if len(smooth_p) > 4:
        chroma_adjust = smooth_p[4]
    else:
        chroma_adjust = 'none'

    # define colormap
    colormap = colormap.lower()
    colormap_enabled = (colormap != "none" and colormap != "")
    if colormap_enabled:
        colormap_adjust = havc_utils._get_colormap(colormap)
    else:
        colormap_adjust = "none"

    # unpack chroma_stabilizer
    stab_enabled = stab
    stab_nframes = stab_p[0]
    stab_mode = stab_p[1]
    stab_sat = stab_p[2]
    stab_tht = stab_p[3]
    stab_weight = stab_p[4]
    stab_tht_scen = stab_p[5]
    if len(stab_p) > 6:
        stab_hue_adjust = stab_p[6]
    else:
        stab_hue_adjust = 'none'
    stab_algo = 0

    clip_colored = clip

    if dark_enabled:
        clip_colored = vs_dark_tweak(clip_colored, dark_threshold=dark_threshold, dark_amount=dark_amount,
                                     dark_hue_adjust=dark_hue_adjust.lower())

    if chroma_smoothing_enabled:
        clip_colored = vs_chroma_bright_tweak(clip_colored, black_threshold=black_threshold,
                                              white_threshold=white_threshold, dark_sat=dark_sat,
                                              dark_bright=dark_bright, chroma_adjust=chroma_adjust.lower())

    if colormap_enabled:
        clip_colored = vs_colormap(clip_colored, colormap=colormap_adjust)

    if stab_enabled:
        clip_colored = vs_chroma_stabilizer_ex(clip_colored, nframes=stab_nframes, mode=stab_mode, sat=stab_sat,
                                               tht=stab_tht, weight=stab_weight, hue_adjust=stab_hue_adjust.lower(),
                                               tht_scen=stab_tht_scen, algo=stab_algo)
        clip_colored = vs_reduce_flicker(clip_colored)

    if chroma_resize_enabled:
        clip_new = _clip_chroma_resize(clip_orig, clip_colored)
    else:
        clip_new = clip_colored

    return restore_format(clip_new, orig_fmt)



"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
Utility function to slices the input clip into 2 or 4 overlapping tiles (2x2 grid).
"""

def HAVC_clip_slice(clip: vs.VideoNode, slices: int = 2, overlap_x: int = 32, overlap_y: int = 32) -> ClipTiles:
    """
    Slices the input clip into 2 or 4 overlapping tiles (2x2 grid).

    Args:
        clip: clip to be sliced in 2 or 4 tiles
        slices: number of slices, allowed values are 2 or 4, default = 2
        overlap_x: horizontal overlap pixels size, default = 32
        overlap_y: vertical overlap pixels size, default = 32 (used only if slices = 4)

    Returns:
        class ClipTiles with items:
            - clip_orig: original clip
            - tiles: [tl, tr, bl, br] — each larger than H/2 x W/2 due to overlap
            - original_width: original width
            - original_height: original height
            - base_tile_w: base tile width (without extra padding)
            - base_tile_h: base tile height (without extra padding)
            - overlap_x: actual horizontal overlap pixels
            - overlap_y: actual vertical overlap pixels
    """

    if slices == 4:
        return vs_slice_into_2x2_overlapping_tiles(clip, overlap_x, overlap_y)
    else:
        return vs_slice_into_2_horizontal_tiles(clip, overlap_x)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
Utility function to reconstructs the original clip from 2 or 4 overlapping tiles.
"""

def HAVC_clip_reconstruct(clip_tiles: ClipTiles, blend_weight: float = 0.5, chroma_resize: bool = False) -> vs.VideoNode:
    """
    Reconstructs the original clip from 4 overlapping filtered tiles,
    using smooth linear blending in overlap zones.

    Args:
        clip_tiles: class ClipTiles with items
            - tiles: list of sliced clips
            - base_tile_w: base tile width (without extra padding)
            - base_tile_h: base tile height (without extra padding)
            - overlap_x: actual horizontal overlap pixels
            - overlap_y: actual vertical overlap pixels
        blend_weight: fixed weight to merge the overlapping region, if == 0 is performed a merge with linear
                      increasing weight from 0 to 1 in the overlapping region for smooth transitions, default = 0.5
        chroma_resize: if True will be performed a chroma Resize. The Y plane of reconstructed clip will be 
                       replaced by the Y plane of the original clip

    Return:
       clip after reconstruction = original size H × W.
    """
    if len(clip_tiles.tiles) == 4:
        return vs_reconstruct_from_2x2_overlapping_tiles(clip_tiles, blend_weight, chroma_resize)
    else:
        return vs_reconstruct_from_2_horizontal_tiles(clip_tiles, blend_weight, chroma_resize)
"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
Utility function to restore the colors of gray pixels.
"""


def HAVC_recover_clip_color(clip: vs.VideoNode = None, clip_color: vs.VideoNode = None, sat: float = 0.8, tht: int = 30,
                            strength: float = 1.0, alpha: float = 2.0, mask_weight: float = 1.0, chroma_resize: bool = True,
                            return_mask: bool = False, binary_mask: bool = False, algo: int = 0) -> vs.VideoNode:
    """Utility function to restore the colors of gray pixels in the input clip by using the colors provided in the clip:
       clip_color. Useful to repair the clips colored with DeepRemaster

        :param clip:          clip to repair the colors, any format is supported
        :param clip_color:    clip with the colors to restore, any format is supported
        :param sat:           this parameter allows to change the saturation of colored clip (default = 0.8)
        :param tht:           threshold to identify gray pixels, range[0, 255] (default = 30)
        :param strength:      represent the strength of the filter. Range[0,1] (default = 1.0)
        :param alpha:         parameter used to control the steepness of gradient curve, values above the default value
                              will preserve more pixels, but could introduce some artifacts, range[1, 10] (default = 2)
        :param mask_weight:   represent the weight of masked colored clip when merged with clip_a. Range[0,1] (default = 1.0)
        :param chroma_resize: if True, the frames will be resized to improve the filter speed (default = True)
        :param return_mask:   if True, will be returned the mask used to identify the gray pixels (white region), could
                              be useful to visualize the gradient mask for debugging, (default = false).
        :param binary_mask:   if True, will be used a binary mask instead of a gradient mask, could be useful to get a
                              clear view on the selected desaturated regions for debugging, (default = false)
        :param algo:          algorithm to build the mask, allowed values are:
                                    [0] = Linear decay with steep gradient, (default)
                                    [1] = Linear decay
                                    [2] = Exponential decay
    """

    clip, orig_fmt = convert_format_RGB24(clip)

    if not isinstance(clip_color, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_merge: this is not a clip: clip_color")

    clip_color, orig_fmt_c = convert_format_RGB24(clip_color)

    clip_restored = ChromaRetentionMerge(clip_a=clip, clip_b=clip_color, sat=sat, tht=tht, clipb_weight=strength,
                                     alpha=alpha, mask_weight=mask_weight, scenechange=False, chroma_resize=chroma_resize,
                                     return_mask=return_mask, binary_mask=binary_mask, algo=algo)

    return restore_format(clip_restored, orig_fmt)


def HAVC_TimeCube(clip: vs.VideoNode, strength: float= 1.0, lut_effect: int = 0, factors: list = None) -> vs.VideoNode:
    """Utility function to apply TimeCube effects

    :param clip:                clip to process, any format is supported.
    :param strength:            strength of the filter, range [0, 1]. Default = 1
    :param lut_effect:          LUT effect to apply, range [0,6], allowed values are:
                                    DEF_LUT_Forest_Film: int = 0
                                    DEF_LUT_City_Skyline: int = 1
                                    DEF_LUT_Exploration: int = 2
                                    DEF_LUT_FUJ_Film: int = 3
                                    DEF_LUT_Hollywood: int = 4
                                    DEF_LUT_Classic_Film: int = 5
                                    DEF_LUT_Warm_Haze: int = 6
                                    DEF_LUT_HDR_Color: int = 7
                                    DEF_LUT_Amber_Light: int = 8
                                    DEF_LUT_Blue_Mist: int = 9
                                    DEF_LUT_Vintage_Fox: int = 10
                                    DEF_LUT_Flat_Pop: int = 11
    :param factors:             if is not set to None, is necessary to provide a list
                                with the full adjustment factors with the following order:
                                    factors[0] -> hue (default = 0)
                                    factors[1] -> sat (default = 1)
                                    factors[2] -> bright (default = 0)
                                    factors[3] -> cont (default = 1)
                                    factors[4] -> gamma (default = 1)
    """

    clip, orig_fmt = convert_format_RGB24(clip)

    clip = vs_timecube(clip, strength, lut_effect, factors)

    return restore_format(clip, orig_fmt)


def HAVC_clip_overlay(
        base: vs.VideoNode,
        overlay: vs.VideoNode,
        x: int = 0,
        y: int = 0,
        mask: vs.VideoNode | None = None,
        opacity: float = 1.0,
        mode: str = 'normal',
        planes: (int | Sequence[int]) | None = None,
        mask_first_plane: bool = True,
) -> vs.VideoNode:
    """
    Puts clip overlay on top of clip base using different blend modes, and with optional x,y positioning, masking and opacity.

    Parameters:
    :param base:      This clip will be the base, determining the size and all other video properties of the result.
    :param overlay:   This is the image that will be placed on top of the base clip.
    :param x, y:      Define the placement of the overlay image on the base clip, in pixels. Can be positive or negative.
    :param mask:      Optional transparency mask. Must be the same size as overlay. Where mask is darker, overlay will be more transparent.
    :param opacity:   Set overlay transparency. The value is from 0.0 to 1.0, where 0.0 is transparent and 1.0 is fully opaque.
                      This value is multiplied by mask luminance to form the final opacity.
    :param mode:      Defines how your overlay should be blended with your base image. Available blend modes are:
                      normal, addition, average, difference, divide, exclusion, multiply, overlay, subtract.
    :param planes:    Specifies which planes will be processed. Any unprocessed planes will be simply copied.
    :param mask_first_plane: If true, only the mask's first plane will be used for transparency.
    """

    if not (isinstance(base, vs.VideoNode) and isinstance(overlay, vs.VideoNode)):
        raise vs.Error('mask_overlay: this is not a clip')

    if mask is not None:
        if not isinstance(mask, vs.VideoNode):
            raise vs.Error('mask_overlay: mask is not a clip')

        if (mask.width != overlay.width or mask.height != overlay.height or
                mask.format.bits_per_sample != overlay.format.bits_per_sample):
            raise vs.Error('mask_overlay: mask must have the same dimensions and bit depth as overlay')

    if base.format.sample_type == vs.INTEGER:
        bits = base.format.bits_per_sample
        neutral = 1 << (bits - 1)
        peak = (1 << bits) - 1
        factor = 1 << bits
    else:
        neutral = 0.5
        peak = factor = 1.0

    plane_range = range(base.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    if base.format.subsampling_w > 0 or base.format.subsampling_h > 0:
        base_orig = base
        base = base.resize.Point(format=base.format.replace(subsampling_w=0, subsampling_h=0))
    else:
        base_orig = None

    if overlay.format.id != base.format.id:
        overlay = overlay.resize.Point(format=base.format)

    if mask is None:
        mask = overlay.std.BlankClip(
            format=overlay.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0), color=peak)
    elif mask.format.id != overlay.format.id and mask.format.color_family != vs.GRAY:
        mask = mask.resize.Point(format=overlay.format, range_s='full')

    opacity = min(max(opacity, 0.0), 1.0)
    mode = mode.lower()

    # Calculate padding sizes
    l, r = x, base.width - overlay.width - x
    t, b = y, base.height - overlay.height - y

    # Split into crop and padding values
    cl, pl = min(l, 0) * -1, max(l, 0)
    cr, pr = min(r, 0) * -1, max(r, 0)
    ct, pt = min(t, 0) * -1, max(t, 0)
    cb, pb = min(b, 0) * -1, max(b, 0)

    # Crop and padding
    overlay = overlay.std.Crop(left=cl, right=cr, top=ct, bottom=cb)
    overlay = overlay.std.AddBorders(left=pl, right=pr, top=pt, bottom=pb)
    mask = mask.std.Crop(left=cl, right=cr, top=ct, bottom=cb)
    mask = mask.std.AddBorders(left=pl, right=pr, top=pt, bottom=pb, color=[0] * mask.format.num_planes)

    if opacity < 1:
        mask = mask.std.Expr(expr=f'x {opacity} *')

    if mode == 'normal':
        pass
    elif mode == 'addition':
        expr = f'x y +'
    elif mode == 'average':
        expr = f'x y + 2 /'
    elif mode == 'difference':
        expr = f'x y - abs'
    elif mode == 'divide':
        expr = f'y 0 <= {peak} {peak} x * y / ?'
    elif mode == 'exclusion':
        expr = f'x y + 2 x * y * {peak} / -'
    elif mode == 'multiply':
        expr = f'x y * {peak} /'
    elif mode == 'overlay':
        expr = f'x {neutral} < 2 x y * {peak} / * {peak} 2 {peak} x - {peak} y - * {peak} / * - ?'
    elif mode == 'subtract':
        expr = f'x y -'
    else:
        raise vs.Error('mask_overlay: invalid mode specified')

    if mode != 'normal':
        overlay = vs.core.std.Expr([overlay, base], expr=[expr if i in planes else '' for i in plane_range])

    # Return padded clip
    last = vs.core.std.MaskedMerge(base, overlay, mask, planes=planes, first_plane=mask_first_plane)
    if base_orig is not None:
        last = last.resize.Point(format=base_orig.format)
    return last

def HAVC_auto_levels(clip: vs.VideoNode = None, mode: str = 'Light', method: int = 0,
                     luma_blend: bool = False, range_tv: bool = True) -> vs.VideoNode:
    """Pre-/Post-process filter using Histogram Equalization or Retinex for improving contrast/luminosity
       of B&W clips to be colored with HAVC.

        :param clip:          Clip to process, any format is supported.
        :param mode:          This parameter allows to improve contrast and luminosity of input clip
                              Allowed values are:
                                  'Light', (default)
                                  'Medium',
                                  'Strong'
        :param method:        Method used to perform the histogram equalization.
                              Allowed values are:
                                0 : CLAHE (luma) (default)
                                1 : Simple Histogram Equalization on all RGB channels
                                2 : CLAHE on all RGB channels
                                3 : CLAHE (luma) + Simple (RGB)
                                4 : ScaleAbs + LUT
                                5 : Multi-Scale Retinex on Luma
        :param luma_blend:    If enabled the equalized image is blended with the original image, darker is the image and more
                              weight will be assigned to the original image, default = True
        :param range_tv:      If True, perform the color adjustments on limited TV range (the filter works better
                              in TV range), default = True
        """

    clip, orig_fmt = convert_format_RGB24(clip)

    clip_new = vs_auto_levels(clip, mode, method, luma_blend, range_tv)

    return restore_format(clip_new, orig_fmt)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function vSceneDetect() to set the scene-change frames in the clip
"""


def HAVC_SceneDetect(clip: vs.VideoNode, sc_threshold: float = constants.DEF_THRESHOLD, sc_tht_offset: int = 1,
                     sc_tht_ssim: float = 0.0, sc_min_int: int = 1, sc_min_freq: int = 0, sc_normalize: bool = False,
                     sc_tht_white: float = constants.DEF_THT_WHITE, sc_tht_black: float = constants.DEF_THT_BLACK,
                     sc_debug: bool = False) -> vs.VideoNode:
    """Utility function to set the scene-change frames in the clip

    :param clip:                clip to process, any format is supported.
    :param sc_threshold:        Scene change threshold used to generate the reference frames.
                                It is a percentage of the luma change between the previous n-frame (n=sc_tht_offset)
                                and the current frame. range [0-1], default 0.10.
    :param sc_tht_offset:       Offset index used for the Scene change detection. The comparison will be performed,
                                between frame[n] and frame[n-offset]. An offset > 1 is useful to detect blended scene
                                change, range[1, 25]. Default = 1.
    :param sc_normalize:        If true the B&W frames are normalized before use misc.SCDetect(), the normalization will
                                increase the sensitivity to smooth scene changes.
    :param sc_tht_white:        Threshold to identify white frames, range [0-1], default 0.85.
    :param sc_tht_black:        Threshold to identify dark frames, range [0-1], default 0.15.
    :param sc_tht_ssim:         Threshold used by the SSIM (Structural Similarity Index Metric) selection filter.
                                If > 0, will be activated a filter that will improve the scene-change detection,
                                by discarding images that are similar.
                                Suggested values are between 0.35 and 0.85, range [0-1], default 0.0 (deactivated)
    :param sc_min_int:          Minimum number of frame interval between scene changes, range[1, 25]. Default = 1.
    :param sc_min_freq:         if > 0 will be generated at least a reference frame every "sc_min_freq" frames.
                                range [0-1500], default: 0.
    :param sc_debug:            Enable SC debug messages. default: False

    """
    clip, orig_fmt = convert_format_RGB24(clip)

    clip = SceneDetect(clip, threshold=sc_threshold, tht_offset=sc_tht_offset, frequency=sc_min_freq,
                       sc_tht_filter=sc_tht_ssim, min_length=sc_min_int, tht_white=sc_tht_white,
                       tht_black=sc_tht_black, frame_norm=sc_normalize, sc_debug=sc_debug)

    return restore_format(clip, orig_fmt)


def HAVC_SceneDetectEdges(clip: vs.VideoNode, sc_threshold: float = 0.035, sc_tht_offset: int = 2,
                     sc_tht_ssim: float = 0.80, sc_min_int: int = 20, sc_mult_tht: int = 15,
                     sc_tht_white: float = 0.70, sc_tht_black: float = 0.10,
                     sc_debug: bool = False) -> vs.VideoNode:
    """Utility function to set the scene-change frames in the clip

    :param clip:                clip to process, any format is supported.
    :param sc_threshold:        Scene change threshold used to generate the reference frames.
                                It is a percentage of the luma change between the previous n-frame edges
                                (n=sc_tht_offset) and the current frame edges. range [0.020-0.090], default 0.035.
    :param sc_tht_offset:       Offset index used for the Scene change detection. The comparison will be performed,
                                between frame[n] and frame[n-offset]. An offset > 1 is useful to detect blended scene
                                change, range[1, 25]. Default = 2.
    :param sc_tht_white:        Threshold to identify white frames, range [0-1], default 0.70.
    :param sc_tht_black:        Threshold to identify dark frames, range [0-1], default 0.10.
    :param sc_tht_ssim:         Threshold used by the SSIM (Structural Similarity Index Metric) selection filter.
                                If > 0, will be activated a filter that will improve the scene-change detection,
                                by discarding images that are similar.
                                Suggested values are between 0.35 and 0.85, range [0-1], default 0.80
    :param sc_min_int:          Minimum number of frame interval between scene changes, range[1, 25]. Default = 20.
    :param sc_mult_tht:         Threshold multiplier used to identify significant scene change, range[5, 25],
                                default = 15
    :param sc_debug:            Enable SC debug messages. default: False

    """
    clip, orig_fmt = convert_format_RGB24(clip)

    clip = SceneDetectEdges(clip, threshold=sc_threshold, ssim_threshold=sc_tht_ssim, sc_diff_offset=sc_tht_offset,
                            sc_min_int=sc_min_int, sc_mult_tht=sc_mult_tht, tht_white=sc_tht_white,
                            tht_black=sc_tht_black, sc_debug=sc_debug)

    return restore_format(clip, orig_fmt)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function SceneDetect and vs_sc_export_frames() to export the clip's 
reference frames
"""


def HAVC_extract_reference_frames(clip: vs.VideoNode, sc_threshold: float = constants.DEF_THRESHOLD, sc_tht_offset: int = 1,
                                  sc_tht_ssim: float = 0.0, sc_min_int: int = 1, sc_min_freq: int = 0,
                                  sc_framedir: str = "./", sc_sequence: bool = False, sc_normalize: bool = False,
                                  ref_offset: int = 0, sc_tht_white: float = constants.DEF_THT_WHITE,
                                  sc_tht_black: float = constants.DEF_THT_BLACK, ref_ext: str = constants.DEF_EXPORT_FORMAT,
                                  ref_jpg_quality: int = constants.DEF_JPG_QUALITY, ref_override: bool = True,
                                  sc_algo:int =0, sc_debug: bool = False) -> vs.VideoNode:
    """Utility function to export reference frames

    :param clip:                clip to process, any format is supported.
    :param sc_threshold:        Scene change threshold used to generate the reference frames.
                                It is a percentage of the luma change between the previous n-frame (n=sc_offset)
                                and the current frame.
                                algo=0: suggested values are between 0.09 and 0.15 (best=0.10).
                                algo=1: suggested values are between 0.06 and 0.09 (best=0.07).
                                range [0-1], default 0.10.
    :param sc_tht_offset:       Offset index used for the Scene change detection. The comparison will be performed,
                                between frame[n] and frame[n-offset]. An offset > 1 is useful to detect blended scene
                                change.
                                algo=0: suggested values are between 1 and 5 (best=1).
                                algo=1: suggested values are between 1 and 5 (best=2).
                                range[1, 25]. Default = 1.
    :param sc_normalize:        If true the B&W frames are normalized before use misc.SCDetect(), the normalization will
                                increase the sensitivity to smooth scene changes (used only by algo=0).
    :param sc_tht_white:        Threshold to identify white frames, range [0-1], default 0.70.
    :param sc_tht_black:        Threshold to identify dark frames, range [0-1], default 0.10.
    :param sc_tht_ssim:         Threshold used by the SSIM (Structural Similarity Index Metric) selection filter.
                                If > 0, will be activated a filter that will improve the scene-change detection,
                                by discarding images that are similar.
                                algo=0: suggested values are between 0.35 and 0.85 (best=0.60).
                                algo=1: suggested values are between 0.10 and 0.20 (best=0.14).
                                range [0-1], default 0.0 (deactivated)
    :param sc_min_int:          Minimum number of frame interval between scene changes, range[1, 50]. Default = 1.
                                algo=0: suggested values are between 1 and 0.25 (best=10).
                                algo=1: suggested values are between 10 and 50 (best=30).
    :param sc_min_freq:         if > 0 will be generated at least a reference frame every "sc_min_freq" frames.
                                range [0-1500], default: 0 (auto).
                                algo=0: suggested values are between 0 and 25 (best=0).
                                algo=1: suggested values are between 5 and 10 (best=7).
    :param sc_framedir:         If set, define the directory where are stored the reference frames.
                                The reference frames are named as: ref_nnnnnn.[jpg|png].
    :param sc_sequence:         If True, the reference frames will be exported in sequence, using consecutive numbers.
    :param ref_offset:          Offset number that will be added to the number of generated frames. default: 0.
    :param ref_ext:             File extension and format of saved frames, range ["jpg", "png"] . default: "jpg"
    :param ref_jpg_quality:     Quality of "jpg" compression, range[0,100]. default: 95
    :param ref_override:        If True, the reference frames with the same name will be overridden, otherwise will
                                be discarded. default: True
    :param sc_algo:             Algorithm applied for scene detection, allowed values are:
                                   0: It will be applied standard SCDetect() method + SSIM detection
                                   1: It will be applied advance detection on the edges.
                                   2: It will be applied SCXvid plugin (very simple)
                                   3: IT will be used SCDetection from MVTools
                                Default = 0
    :param sc_debug:            Enable SC debug messages. default: False

    """
    clip, orig_fmt = convert_format_RGB24(clip)

    pathlib.Path(sc_framedir).mkdir(parents=True, exist_ok=True)

    if sc_algo == 0:
        clip = SceneDetect(clip, threshold=sc_threshold, tht_offset=sc_tht_offset, frequency=sc_min_freq,
                           sc_tht_filter=sc_tht_ssim, min_length=sc_min_int, tht_white=sc_tht_white,
                           tht_black=sc_tht_black, frame_norm=sc_normalize, sc_debug=sc_debug)
    elif sc_algo == 1:
        clip = SceneDetectEdges(clip, threshold=sc_threshold, ssim_threshold=sc_tht_ssim, sc_diff_offset=sc_tht_offset,
                                sc_min_int=sc_min_int, sc_mult_tht=sc_min_freq, tht_white=sc_tht_white,
                                tht_black=sc_tht_black, sc_debug=sc_debug)
    elif sc_algo == 2:
        clip = vs_sc_xvid(clip, use_slices=True)

    else:
        thscd1 = min(int(sc_threshold*2500), 1000)
        thscd2 = min(int(sc_tht_ssim*300), 300)
        clip = vs_mv_sc_detect(clip, thscd1=thscd1, thscd2=thscd2)

    clip = vs_sc_export_frames(clip, sc_framedir=sc_framedir, ref_offset=ref_offset, ref_ext=ref_ext,
                               ref_jpg_quality=ref_jpg_quality, ref_override=ref_override, sequence=sc_sequence)

    return restore_format(clip, orig_fmt)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function vs_sc_export_frames() to export the clip's reference frames
"""


def HAVC_export_reference_frames(clip: vs.VideoNode, sc_framedir: str = "./", ref_offset: int = 0,
                                 ref_ext: str = constants.DEF_EXPORT_FORMAT, ref_jpg_quality: int = constants.DEF_JPG_QUALITY,
                                 ref_override: bool = True) -> vs.VideoNode:
    """Utility function to export reference frames

    :param clip:                clip to process, any format is supported.
    :param sc_framedir:         If set, define the directory where are stored the reference frames.
                                The reference frames are named as: ref_nnnnnn.[jpg|png].
    :param ref_offset:          Offset number that will be added to the number of generated frames. default: 0.
    :param ref_ext:             File extension and format of saved frames, range ["jpg", "png"] . default: "jpg"
    :param ref_jpg_quality:     Quality of "jpg" compression, range[0,100]. default: 95
    :param ref_override:        If True, the reference frames with the same name will be overridden, otherwise will
                                be discarded. default: True
    """

    clip, orig_fmt = convert_format_RGB24(clip)

    pathlib.Path(sc_framedir).mkdir(parents=True, exist_ok=True)
    clip = vs_sc_export_frames(clip, sc_framedir=sc_framedir, ref_offset=ref_offset, ref_ext=ref_ext,
                               ref_jpg_quality=ref_jpg_quality, ref_override=ref_override)

    return restore_format(clip, orig_fmt)

def HAVC_export_list_frames(clip: vs.VideoNode, sc_framedir: str = "./", ref_list: list[int] | None= None,
                            offset: int = 0, ref_ext: str = constants.DEF_EXPORT_FORMAT, ref_jpg_quality: int = constants.DEF_JPG_QUALITY,
                            ref_override: bool = True, fast_extract: bool = True) -> vs.VideoNode:
    """Utility function to export frames included in a list.

    :param clip:                clip to process, any format is supported.
    :param sc_framedir:         If set, define the directory where are stored the frames.
                                The frames are named as: ref_nnnnnn.[jpg|png].
    :param ref_list:            List of frame numbers to export. default: None. If ref_list contains only one frame
                                number, for example ref_list = [25], will be exported a frame every 25 frames
    :param offset:              The offset will be added to the frame number. default = 0.
    :param ref_ext:             File extension and format of saved frames, range ["jpg", "png"] . default: "jpg"
    :param ref_jpg_quality:     Quality of "jpg" compression, range[0,100]. default: 95
    :param ref_override:        If True, the frames with the same name will be overridden, otherwise will
                                be discarded. default: True
    :param fast_extract:        If True, the frames will be extracted directly with get_frame(), otherwise will
                                be performed a full parsing of the clip (necessary if there is a sequential temporal
                                dependency in the script calling this function). default = True
    """
    if ref_list is None or len(ref_list) < 1:
        return clip

    clip, orig_fmt = convert_format_RGB24(clip)

    pathlib.Path(sc_framedir).mkdir(parents=True, exist_ok=True)

    clip = vs_list_export_frames(clip, sc_framedir=sc_framedir, ref_list=ref_list, ref_ext=ref_ext, offset=offset,
                                 ref_jpg_quality=ref_jpg_quality, ref_override=ref_override,fast_extract=fast_extract)

    return restore_format(clip, orig_fmt)

def HAVC_set_tweak_params(tweaks_param: list = None):
    """Utility function to set ddcolor tweak parameters

    :param tweaks_param:       list of values for ddcolor tweak parameters.
                               The template list for tweak parameters the following:
                                    tweaks_param[0] = bright
                                    tweaks_param[1] = cont
                                    tweaks_param[2] = gamma
                                    tweaks_param[3] = luma_constrained_tweak
                                    tweaks_param[4] = luma_min
                                    tweaks_param[5] = gamma_luma_min
                                    tweaks_param[6] = gamma_alpha
                                    tweaks_param[7] = gamma_min
    """

    if tweaks_param is None:
        return

    #global DEF_TWEAK_p

    constants.DEF_TWEAK_p = tweaks_param.copy()

def HAVC_set_debug_level(debug_level: int = 0):
    """Utility function to set HAVC debug level

    :param debug_level:       list of values for HAVC debug level.
                              Allowed values are the following:
                                    DEF_LEVEL_NONE = 0 -> disable any message
                                    DEF_LEVEL_INFO = 1 -> info messages are enabled
                                    DEF_LEVEL_DEBUG = 3 -> info & debug messages are enabled
    """

    if debug_level in (constants.DEF_LEVEL_NONE, constants.DEF_LEVEL_INFO, constants.DEF_LEVEL_DEBUG):
        constants.DEF_DEBUG_LEVEL = debug_level

def HAVC_set_merge_params(method: int = 2, merge_params: list = None):
    """Utility function to set the combination parameters

    :param method:             method used to combine clipa with clipb (default = 2):
                                    0   : clipa only (no merge)
                                    1   : clipb only (no merge)
                                    2   : Simple Merge (default)
                                    3,7 : Constrained Chroma Merge: list[float, bool, int, int]
                                    4   : Luma Masked Merge: list[float, float, float]
                                    5   : Adaptive Luma Merge: list[float, float, float]
                                    6   : Chroma Retention Merge: list[float, float, float, bool, float, int]
    :param merge_params:       list of values for the selected method (no check is performed on values and
                               number of parameters in the list). The list to be passed depend on selected method.
                               The template list for each method is the following:
                                    3,7 : Constrained Chroma Merge: list[float, bool, int, int]
                                            [0] chroma_threshold (default = 0.15)
                                            [1] red_fix (default = True)
                                            [2] base_tol (default = 20)
                                            [3] max_extra (default = 24)
                                    4  : Luma Masked Merge: list[float, float, float]
                                            [0] luma_mask_limit (default = 0.15)
                                            [1] luma_white_limit (default = 0.65)
                                            [2] luma_mask_sat (default = 1.0)
                                    5  : Adaptive Luma Merge: list[float, float, float]
                                            [0] luma_threshold (default = 0.8)
                                            [1] alpha (default = 1.0)
                                            [2] min_weight (default = 0.15)
                                    6  : Chroma Retention Merge: list[float, float, float, bool, float, int]
                                            [0] sat (default = 0.8)
                                            [1] threshold (default = 30)
                                            [2] alpha (default = 2.0)
                                            [3] resize (default = False)
                                            [4] mask_weight (default = 0)
                                            [5] algo (default = 0)
    """

    if merge_params is None or method in (0, 1, 2):
        return

    match method:
        case 3:
            constants.DEF_CMC_p = merge_params.copy()
        case 4:
            constants.DEF_LMM_p = merge_params.copy()
        case 5:
            constants.DEF_ALM_p = merge_params.copy()
        case 6:
            constants.DEF_CRT_p = merge_params.copy()
        case 7:
            constants.DEF_CMC_p = merge_params.copy()
        case _:
            # handle invalid method
            HAVC_LogMessage(MessageType.EXCEPTION, f"HAVC_set_merge_params: Unsupported method: {method}")

"""
------------------------------------------------------------------------------------------------------------------------ 
                                   HAVC INTERNAL FUNCTIONS
------------------------------------------------------------------------------------------------------------------------ 
"""

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function vs_sc_export_frames() to export the clip's reference frames
"""


def _extract_reference_frames(clip: vs.VideoNode, sc_framedir: str = "./", ref_offset: int = 0, ref_ext: str = "png",
                              ref_override: bool = True, prop_name: str = "_SceneChangePrev") -> vs.VideoNode:
    """Export scene-change frames from clip to a directory on disk.

    Creates sc_framedir if it does not exist, converts the clip to RGB24, then delegates
    to vs_sc_export_frames. Primarily used by HAVC_main to persist reference frames for
    subsequent exemplar-based passes.

    :param clip:         Input clip (any format).
    :param sc_framedir:  Output directory for reference images. Created if missing.
    :param ref_offset:   Value added to the frame number in each filename. Default 0.
    :param ref_ext:      Image format extension ('png' or 'jpg'). Default 'png'.
    :param ref_override: If False, existing files are not overwritten. Default True.
    :param prop_name:    Frame property used to detect scene changes. Default '_SceneChangePrev'.
    :return:             Clip pass-through (side-effect: reference frames saved to disk).
    """
    pathlib.Path(sc_framedir).mkdir(parents=True, exist_ok=True)

    clip, orig_fmt = convert_format_RGB24(clip)

    clip = vs_sc_export_frames(clip, sc_framedir=sc_framedir, ref_offset=ref_offset, ref_ext=ref_ext,
                               ref_override=ref_override, prop_name=prop_name)
    return restore_format(clip, orig_fmt)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function vs_recover_clip_luma().
"""


def _clip_chroma_resize(clip_hires: vs.VideoNode, clip_lowres: vs.VideoNode) -> vs.VideoNode:
    """Upscale clip_lowres to the dimensions of clip_hires and replace its luma with the hi-res luma.

    :param clip_hires:  High-resolution clip whose luma plane will be preserved.
    :param clip_lowres: Low-resolution clip whose chroma planes will be upscaled and blended in.
    :return:            Clip at clip_hires resolution with luma from clip_hires and chroma from the upscaled clip_lowres.
    """
    clip_resized = clip_lowres.resize.Spline64(width=clip_hires.width, height=clip_hires.height)

    clip_hires, orig_fmt_h = convert_format_RGB24(clip_hires)
    clip_resized, orig_fmt_r = convert_format_RGB24(clip_resized)

    clip_recovered = vs_recover_clip_luma(clip_hires, clip_resized)

    return restore_format(clip_recovered, orig_fmt_h)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function vs_get_clip_frame() to get frames fast.
"""


def _get_clip_frame(clip: vs.VideoNode, nframe: int = 0) -> vs.VideoNode:
    """Extract a single frame from the clip and return it as a one-frame clip, preserving the original format.

    :param clip:    Input clip, any format is supported.
    :param nframe:  Zero-based index of the frame to extract. Default = 0.
    :return:        Single-frame clip containing the requested frame in the original clip format.
    """
    clip, orig_fmt = convert_format_RGB24(clip)
    clip = vs_get_clip_frame(clip=clip, nframe=nframe)
    return restore_format(clip, orig_fmt)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
disable packages warnings.
"""


def disable_warnings():
    """Suppress noisy log output from third-party libraries used by HAVC.

    Sets the log level to ERROR for known verbose packages (matplotlib, PIL, torch,
    numpy, tensorrt, kornia, dinov2) and silences FutureWarning, UserWarning, and
    DeprecationWarning categories project-wide.
    """
    logger_blocklist = [
        "matplotlib",
        "PIL",
        "torch",
        "numpy",
        "tensorrt",
        "torch_tensorrt"
        "kornia",
        "dinov2"  # dinov2 is issuing warnings not allowing ColorMNetServer to work properly
    ]

    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.ERROR)

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    # warnings.simplefilter(action="ignore", category=Warning)

    torch._logging.set_logs(all=logging.ERROR)

"""
------------------------------------------------------------------------------------------------------------------------ 
                                   DDEOLDIFY LEGACY FUNCTIONS (deprecated)
------------------------------------------------------------------------------------------------------------------------ 
"""


def HAVC_ddeoldify(
        clip: vs.VideoNode, method: int = 2, mweight: float = 0.4, deoldify_p: list = (0, 24, 1.0, 0.0),
        ddcolor_p: list = (1, 24, 1.0, 0.0, True), ddtweak: bool = False,
        ddtweak_p: list = (constants.DEF_TWEAK_p, "300:360|0.8,0.1"),
        cmc_tresh: float = 0.2, lmm_p: list = (0.2, 0.8, 1.0), alm_p: list = (0.8, 1.0, 0.15), cmb_sw: bool = False,
        sc_threshold: float = 0.0, sc_tht_offset: int = 1, sc_min_freq: int = 0, sc_tht_ssim: float = 0.0,
        sc_normalize: bool = False, sc_min_int: int = 1, sc_tht_white: float = constants.DEF_THT_WHITE,
        sc_tht_black: float = constants.DEF_THT_BLACK, device_index: int = 0, torch_dir: str = model_dir,
        sc_debug: bool = False) -> vs.VideoNode:
    """Deprecated alias for HAVC_colorizer. Use HAVC_colorizer instead.

    Emits a VapourSynth WARNING message on every call and forwards all arguments
    to HAVC_colorizer with the appropriate parameter mapping.
    """
    vs.core.log_message(
        vs.MESSAGE_TYPE_WARNING,
        "Warning: HAVC_ddeoldify is deprecated and may be removed in the future, please use 'HAVC_colorizer' instead.")

    debug_level = constants.DEF_LEVEL_DEBUG if sc_debug else constants.DEF_LEVEL_NONE
    return HAVC_colorizer(clip, method, mweight, deoldify_p, ddcolor_p, [ddtweak, False, False], ddtweak_p, [cmc_tresh], lmm_p, alm_p,
                          constants.DEF_CRT_p, cmb_sw, sc_threshold, sc_tht_offset, sc_min_freq, sc_tht_ssim, sc_normalize, sc_min_int,
                          sc_tht_white, sc_tht_black, device_index, torch_dir, debug_level)


def ddeoldify_main(clip: vs.VideoNode, Preset: str = 'Fast', VideoTune: str = 'Stable', ColorFix: str = 'Violet/Red',
                   ColorTune: str = 'Light', ColorMap: str = 'None', degrain_strength: int = 0,
                   enable_fp16: bool = True) -> vs.VideoNode:
    """Deprecated alias for HAVC_main. Use HAVC_main instead.

    Emits a VapourSynth WARNING message on every call and forwards the core preset/tune
    arguments to HAVC_main. The degrain_strength parameter is ignored.
    """
    vs.core.log_message(
        vs.MESSAGE_TYPE_WARNING,
        "Warning: ddeoldify_main is deprecated and may be removed in the future, please use 'HAVC_main' instead.")

    return HAVC_main(clip=clip, Preset=Preset, VideoTune=VideoTune, ColorFix=ColorFix, ColorTune=ColorTune,
                     ColorMap=ColorMap, enable_fp16=enable_fp16)


def ddeoldify(clip: vs.VideoNode, method: int = 2, mweight: float = 0.4, deoldify_p: list = (0, 24, 1.0, 0.0),
              ddcolor_p: list = (1, 24, 1.0, 0.0, True), dotweak: bool = False, dotweak_p: list = (0.0, 1.0, 1.0, False, 0.2, 0.5, 1.5, 0.5),
              ddtweak: bool = False, ddtweak_p: list = (constants.DEF_TWEAK_p, "300:360|0.8,0.1"),
              degrain_strength: int = 0, cmc_tresh: float = 0.2, lmm_p: list = (0.2, 0.8, 1.0),
              alm_p: list = (0.8, 1.0, 0.15), cmb_sw: bool = False, device_index: int = 0,
              torch_dir: str = model_dir) -> vs.VideoNode:
    """Deprecated alias for HAVC_colorizer. Use HAVC_colorizer instead.

    Emits a VapourSynth WARNING message on every call. The dotweak, dotweak_p, and
    degrain_strength parameters are silently ignored; all other arguments are forwarded
    to HAVC_colorizer with the appropriate parameter mapping.
    """
    vs.core.log_message(
        vs.MESSAGE_TYPE_WARNING,
        "Warning: ddeoldify is deprecated and may be removed in the future, please use 'HAVC_colorizer' instead.")

    return HAVC_colorizer(clip, method, mweight, deoldify_p, ddcolor_p, [ddtweak, False, False], ddtweak_p, [cmc_tresh], lmm_p, alm_p,
                          constants.DEF_CRT_p, cmb_sw, sc_threshold=0, sc_min_freq=0, device_index=device_index, torch_dir=torch_dir)


def ddeoldify_stabilizer(clip: vs.VideoNode, dark: bool = False, dark_p: list = (0.2, 0.8), smooth: bool = False,
                         smooth_p: list = (0.3, 0.7, 0.9, 0.0, "none"),
                         stab: bool = False, stab_p: list = (5, 'A', 1, 15, 0.2, 0.80), colormap: str = "none",
                         render_factor: int = 24) -> vs.VideoNode:
    """Deprecated alias for HAVC_stabilizer. Use HAVC_stabilizer instead.

    Emits a VapourSynth WARNING message on every call and forwards all arguments
    unchanged to HAVC_stabilizer.
    """
    vs.core.log_message(
        vs.MESSAGE_TYPE_WARNING,
        "Warning: ddeoldify_stabilizer is deprecated and may be removed in the future, please use 'HAVC_stabilizer'.")

    return HAVC_stabilizer(clip, dark, dark_p, smooth, smooth_p, stab, stab_p, colormap, render_factor)
