"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2025-02-06
version: 
LastEditors: Dan64
LastEditTime: 2026-05-08
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
HAVC utility filter functions
"""
from __future__ import annotations
import vapoursynth as vs
from functools import partial
import os
import math
import cv2
from PIL import Image
import numpy as np
from typing import NamedTuple


from vshavc.vsslib.vsplugins import load_LSMASHSource_plugin, vs_timecube
import vshavc.vsslib.restcolor as restcolor
from vshavc.vsslib.imfilters import get_image_luma, image_luma_blend
from vshavc.vsslib.vsretinex import vs_retinex

import vshavc.vsslib.vsresize as vsresize

from vshavc.vsslib.constants import *

from vshavc.vsslib.vsutils import HAVC_LogMessage, MessageType, frame_to_image, image_to_frame


# Using NamedTuple for better compatibility with VapourSynth's typical usage patterns
class ClipInfo(NamedTuple):
    clip_orig: vs.VideoNode | None
    format_id: int
    color_family: vs.ColorFamily
    bits_per_sample: int
    matrix: int | None
    color_range: int | None
    chroma_resize: bool

VIDEO_EXTENSIONS = ['.mpg', '.mp4', '.m4v', '.avi', '.mkv', '.mpeg']

# map integer _Matrix to zimg string
MATRIX_INT_TO_STR = {
    0: "rgb",
    1: "709",
    4: "fcc",
    5: "470bg",
    6: "170m",
    7: "240m",
    8: "ycgco",
    9: "2020ncl",
    10: "2020cl",
}

def _get_matrix_str(clip: vs.VideoNode, default: str = "709") -> str:
    """Read _Matrix from frame props and return the equivalent zimg string."""
    if _matrixIsInvalid(clip):
        return default
    matrix_val = clip.get_frame(0).props.get('_Matrix')
    return MATRIX_INT_TO_STR.get(int(matrix_val), default)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
function to convert video clip to RGB24 format and to restore the original format
"""

def convert_format_RGB24(clip: vs.VideoNode, chroma_resize: bool = False) -> tuple[vs.VideoNode, ClipInfo]:
    """
    Convert any clip to RGB24 (8-bit full-range RGB).
    Returns: ClipInfo
    """

    # Store original clip information before any processing
    original_format = clip.format
    if original_format is None:
        HAVC_LogMessage(MessageType.EXCEPTION,"Clip must have a defined format")

    if not isinstance(clip, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "convert_format_RGB24: Input is not a valid clip.")

    # It is assumed that the clip was converted to RGB by HAVC_read_video()
    if clip.format.id == vs.RGB24:
        if vs.core.core_version.release_major < 74:
            clip_color_range = vs.ColorRange(vs.RANGE_FULL)
        else:
            clip_color_range = vs.Range(vs.RANGE_FULL)

        clip_info = ClipInfo(
            clip_orig=clip if chroma_resize else None,
            format_id=original_format.id,
            color_family=original_format.color_family,
            bits_per_sample=original_format.bits_per_sample,
            matrix=vs.MatrixCoefficients(vs.MATRIX_RGB),
            color_range=clip_color_range,
            chroma_resize=chroma_resize,
        )
        if chroma_resize:
            clip = vsresize.resize_min_HW(clip)
        return clip, clip_info

    # Not Clip RGB
    # Set missing color properties to reasonable defaults
    if _matrixIsInvalid(clip):
        clip = clip.std.SetFrameProps(_Matrix=vs.MATRIX_BT709)
    if _rangeIsInvalid(clip):
        if vs.core.core_version.release_major < 74:
            clip = clip.std.SetFrameProps(_ColorRange=vs.RANGE_LIMITED)
        else:
            clip = clip.std.SetFrameProps(_Range=vs.RANGE_LIMITED)
    # : Get frame properties for matrix and color range
    frame = clip.get_frame(0)
    props = frame.props

    if vs.core.core_version.release_major < 74:
        clip_color_range = vs.ColorRange(props.get('_ColorRange', vs.RANGE_LIMITED.value))
    else:
        clip_color_range = vs.Range(props.get('_Range', vs.RANGE_LIMITED.value))

    # resize to max allowed width
    clip_info = ClipInfo(
        clip_orig=clip if chroma_resize else None,
        format_id=original_format.id,
        color_family=original_format.color_family,
        bits_per_sample=original_format.bits_per_sample,
        matrix=vs.MatrixCoefficients(props.get('_Matrix', vs.MATRIX_BT709.value)),
        color_range=clip_color_range,
        chroma_resize=chroma_resize,
    )
    if chroma_resize:
        clip = vsresize.resize_min_HW(clip)

    # Ensure we're working with 8-bit
    if clip.format.bits_per_sample != 8:
        clip = vs.core.resize.Bicubic(clip, format=clip.format.replace(bits_per_sample=8))

    # Convert based on color family
    if original_format.color_family == vs.YUV:
        matrix_val = int(clip.get_frame(0).props.get('_Matrix', 1))
        matrix_str = MATRIX_INT_TO_STR.get(matrix_val, "709")
        # Use detected or default matrix
        clip = vs.core.resize.Bicubic(
            clip,
            format=vs.RGB24,
            matrix_in_s=matrix_str,
            range_in_s="limited",
            range_s="full",
            dither_type="error_diffusion"
        )
    elif original_format.color_family == vs.GRAY:
        clip = vs.core.resize.Bicubic(
            clip,
            format=vs.RGB24,
            range_in_s="limited",  # GRAY is usually limited
            range_s="full"
        )
    else:  # Already RGB but not RGB24 (e.g., RGB48)
        clip = vs.core.resize.Bicubic(
            clip,
            format=vs.RGB24,
            range_s="full"
        )
    clip = clip.std.SetFrameProps(_Matrix=vs.MATRIX_RGB)

    # Ensure output is explicitly full-range
    if vs.core.core_version.release_major < 74:
        clip = clip.std.SetFrameProps(_ColorRange=vs.RANGE_FULL)
    else:
        clip = clip.std.SetFrameProps(_Range=vs.RANGE_FULL)

    return clip, clip_info


def restore_format(clip: vs.VideoNode, clip_info: ClipInfo) -> vs.VideoNode:
    """
    Restore colorized RGB24 clip to a format suitable for the original input.
    - If original was GRAY, output YUV420P8 (8-bit color).
    - If original was YUV, restore to original YUV format.
    - If original was RGB, restore to original RGB format.
    Assumes input 'clip' is full-range RGB24

       :param clip:        clip to process, any format is supported.
       :param clip_info:   ClipInfo struct containing original clip information
    """
    if not isinstance(clip, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "restore_format: Input is not a valid clip.")

    if clip.format.id != vs.RGB24:
        HAVC_LogMessage(MessageType.EXCEPTION, "restore_format: Input clip must be RGB24.")

    if clip_info.chroma_resize:
        clip = vsresize.resize_to_chroma(clip_info.clip_orig, clip)

    # If already in target format (unlikely post-colorization), return as-is
    if clip.format.id == clip_info.format_id:
        return clip

    if clip_info.color_family == vs.YUV:
        # Use the original matrix if available, otherwise default to BT.709
        matrix = clip_info.matrix if clip_info.matrix is not None else vs.MATRIX_BT709
        # Use the original color range if available, otherwise default to limited for YUV
        range_s = "limited"  # YUV is typically limited range by default
        if clip_info.color_range is not None:
            range_s = "full" if clip_info.color_range == vs.RANGE_FULL else "limited"

        restored = vs.core.resize.Bicubic(
            clip,
            format=clip_info.format_id,
            matrix_in=vs.MATRIX_RGB,
            matrix=matrix,
            range_in_s="full",
            range_s=range_s,
            dither_type="error_diffusion"
        )
    elif clip_info.color_family == vs.GRAY:
        # Original was grayscale → output standard 8-bit YUV (colorized result)
        # Use the original color range if available, otherwise default to limited
        range_s = "limited"
        if clip_info.color_range is not None:
            range_s = "full" if clip_info.color_range == vs.RANGE_FULL else "limited"

        restored = vs.core.resize.Bicubic(
            clip,
            format=vs.YUV420P8,
            matrix=vs.MATRIX_BT709,
            range_in_s="full",
            range_s=range_s,
            dither_type="error_diffusion"
        )
    else:
        # Original was RGB (but not RGB24, e.g., RGB48)
        # Use the original color range if available, otherwise default to full for RGB
        range_s = "full"  # RGB is typically full range by default
        if clip_info.color_range is not None:
            range_s = "full" if clip_info.color_range == vs.RANGE_FULL else "limited"

        restored = vs.core.resize.Bicubic(
            clip,
            format=clip_info.format_id,
            range_in_s="full",
            range_s=range_s
        )

    return restored

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
function to read a video clip
"""

def HAVC_read_video(source: str, fpsnum: int = 0, fpsden: int = 1, width: int = 0, height: int = 0,
                    return_rgb: bool = True) -> vs.VideoNode:
    """HAVC utility function to read a video provided externally.
       The clip provided in output will be already in RGB24 format

    :param source:       Full path to the video to read
    :param fpsnum:       FPS numerator, for using it in HAVC, must be provided the
                         same value of clip to be colored: clip.fps_num
    :param fpsden:       FPS denominator, for using it in HAVC, must be provided the
                         same value of clip to be colored: clip.fps_den
    :param width:        If > 0 the clip is resized to width using Spline36
    :param height:       If > 0 the clip is resized to height using Spline36
    :param return_rgb:   If True (default) the clip will be converted in RGB24 format
    :return:             clip in RGB24 format if return_rgb=True
    """

    if not os.path.isfile(source):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC: invalid clip -> " + source)

    ext = source.lower()
    if not any(ext.endswith(extension) for extension in VIDEO_EXTENSIONS):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC: invalid clip extension -> " + source)

    load_LSMASHSource_plugin()

    try:
        clip = vs.core.lsmas.LWLibavSource(source=source, stream_index=0, fpsnum=fpsnum, fpsden=fpsden,
                                           cache=0, prefer_hw=0)
    except Exception as error:
        clip = None
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC: LSMASHSource.dll not loaded or invalid clip -> " + str(error))

    if width > 0 and height > 0:
        clip = clip.resize.Spline36(width=width, height=height)
    elif width > 0:
        clip = clip.resize.Spline36(width=width, height=clip.height)
    elif height > 0:
        clip = clip.resize.Spline36(width=clip.width, height=height)

    # setting color matrix to 709.
    if _matrixIsInvalid(clip):
        clip = vs.core.std.SetFrameProps(clip, _Matrix=vs.MATRIX_BT709)
    # setting color transfer (vs.TRANSFER_BT709), if it is not set.
    if _transferIsInvalid(clip):
        clip = vs.core.std.SetFrameProps(clip=clip, _Transfer=vs.TRANSFER_BT709)
    # setting color primaries info (to vs.PRIMARIES_BT709), if it is not set.
    if _primariesIsInvalid(clip):
        clip = vs.core.std.SetFrameProps(clip=clip, _Primaries=vs.PRIMARIES_BT709)

    # setting color range to TV (limited) range.
    if vs.core.core_version.release_major < 74:
        clip = vs.core.std.SetFrameProps(clip=clip, _ColorRange=vs.RANGE_LIMITED)
    else:
        clip = vs.core.std.SetFrameProps(clip=clip, _Range=vs.RANGE_LIMITED)
    # making sure frame rate is set
    clip = vs.core.std.AssumeFPS(clip=clip, fpsnum=clip.fps_num, fpsden=clip.fps_den)
    # making sure the detected scan type is set (detected: progressive)
    clip = vs.core.std.SetFrameProps(clip=clip, _FieldBased=vs.FIELD_PROGRESSIVE)  # progressive

    if return_rgb:
        # adjusting color space to RGB24 for HAVC
        matrix_str = _get_matrix_str(clip, default="709")
        clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s=matrix_str, range_s="full")
        # adjust _Matrix to RGB
        clip = vs.core.std.SetFrameProps(clip=clip, _Matrix=vs.MATRIX_RGB)

        # changing range from limited to full range for HAVC
        clip = vs.core.resize.Bicubic(clip, range_in_s="limited", range_s="full")
        # setting color range to PC (full) range.
        if vs.core.core_version.release_major < 74:
            clip = vs.core.std.SetFrameProps(clip=clip, _ColorRange=vs.RANGE_FULL)
        else:
            clip = vs.core.std.SetFrameProps(clip=clip, _Range=vs.RANGE_FULL)
    else:
        # setting color range to TV (limited) range.
        if vs.core.core_version.release_major < 74:
            clip = vs.core.std.SetFrameProps(clip=clip, _ColorRange=vs.RANGE_LIMITED)
        else:
            clip = vs.core.std.SetFrameProps(clip=clip, _Range=vs.RANGE_LIMITED)

    return clip


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
Utility functions for HAVC_main
"""


def _get_render_factors(Preset: str) -> tuple[int, int, int]:
    # Select presets / tuning
    Preset = Preset.lower()
    presets = ['placebo', 'veryslow', 'slower', 'slow', 'medium', 'fast', 'faster', 'veryfast']
    preset0_rf = [36, 34, 32, 28, 24, 22, 20, 16]
    preset1_rf = [36, 34, 32, 28, 24, 22, 20, 16]

    pr_id = 5  # default 'fast'
    try:
        pr_id = presets.index(Preset)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: Preset choice is invalid for '" + str(pr_id) + "'")

    return pr_id, preset0_rf[pr_id], preset1_rf[pr_id]


def _get_mweight(VideoTune: str) -> float:
    # Select VideoTune
    VideoTune = VideoTune.lower()
    video_tune = ['verystable', 'morestable', 'stable', 'balanced', 'vivid', 'morevivid', 'veryvivid']
    ddcolor_weight = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    w_id = 3
    try:
        w_id = video_tune.index(VideoTune)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: VideoTune choice is invalid for '" + VideoTune + "'")

    return ddcolor_weight[w_id]


def _get_comb_method(CombMethod: str) -> int:
    # Select VideoTune
    CombMethod = CombMethod.lower()
    comb_str = ['simple', 'constrained-chroma', 'luma-masked', 'adaptive-luma', 'chroma-retention', 'chromabound adaptive']
    method_id = [2, 3, 4, 5, 6, 7]

    w_id = 2
    try:
        w_id = comb_str.index(CombMethod)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: CombMethod choice is invalid for '" + CombMethod + "'")

    return method_id[w_id]

def _spit_color_model(ColorModel: str) -> tuple[str, str]:
    ColorModel = ColorModel.lower()

    deoldify = "none"
    ddcolor = "none"

    if '+' not in ColorModel:
        if 'deoldify' in ColorModel:
            return ColorModel, ddcolor
        else:
            return deoldify, ColorModel

    cm = ColorModel.split("+")

    deoldify = f"deoldify({cm[0]})"

    if cm[1] in ('siggraph17', 'eccv16'):
        ddcolor = f"zhang({cm[1]})"
    else:
        ddcolor = f"ddcolor({cm[1]})"

    return deoldify, ddcolor

def _get_color_model(ColorModel: str) -> tuple[int, int, int]:
    ColorModel = ColorModel.lower()

    ddcolor_list = ["modelscope", "artistic", "siggraph17", "eccv16"]
    deoldify_list = ["video", "stable", "artistic"]

    if '+' in ColorModel:
        cm = ColorModel.split("+")
        do_model = deoldify_list.index(cm[0])
        dd_model = ddcolor_list.index(cm[1])
        dd_method = 2
        return do_model, dd_model, dd_method

    do_model = 0  # default: Video
    dd_model = 0  # default: Modelscope
    cmodel = ""

    if "deoldify" in ColorModel:
        cmodel = ColorModel.replace("deoldify", "").replace("(", "").replace(")", "")
        do_model = deoldify_list.index(cmodel)
        dd_method = 0
        return do_model, dd_model, dd_method

    if "ddcolor" in ColorModel:
        cmodel = ColorModel.replace("ddcolor", "").replace("(", "").replace(")", "")
    elif "zhang" in ColorModel:
        cmodel = ColorModel.replace("zhang", "").replace("(", "").replace(")", "")
    else:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: ColorModel choice is invalid for '" + ColorModel + "'")

    dd_method = 1
    dd_model = ddcolor_list.index(cmodel)

    return do_model, dd_model, dd_method


def _get_temp_color(ColorTemp: str) -> int:
    # Select ColorTemp
    if ColorTemp is None:
        ColorTemp = "none"
    ColorTemp = ColorTemp.lower().replace(" ", "")
    color_temp = ['none', 'veryhigh', 'high', 'medium', 'low', 'verylow']

    ct_id = color_temp.index(ColorTemp)

    return ct_id


def _get_color_tune(ColorTune: str, ColorFix: str, ColorMap: str,
                    dd_model: int) -> tuple[list[bool], str, str, str, str]:

    # set defaults
    dd_tweak = [False, False, False]   # tweaks_enabled, denoise_enabled, retinex_enabled
    hue_range = ""
    hue_range2 = ""
    chroma_adjust = ""
    chroma_adjust2 = ""

    # Select ColorTune for ColorFix
    if ColorTune is None:
        ColorTune = "none"
    ColorTune = ColorTune.lower()
    color_tune = ['none', 'light', 'medium', 'strong']
    if dd_model == 0:
        hue_tune = ["1.0,0.0", "0.7,0.1", "0.5,0.1", "0.2,0.1"]
    elif dd_model == 2:
        hue_tune = ["1.0,0.0", "0.6,0.1", "0.4,0.2", "0.2,0.1"]
    elif dd_model == 3:
        hue_tune = ["1.0,0.0", "0.7,0.1", "0.6,0.1", "0.3,0.1"]
    else:
        hue_tune = ["1.0,0.0", "0.8,0.1", "0.5,0.1", "0.2,0.1"]
    hue_tune2 = ["1.0,0.0", "0.9,0", "0.7,0", "0.5,0"]

    tn_id = 0
    try:
        tn_id = color_tune.index(ColorTune)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: ColorTune choice is invalid for '" + ColorTune + "'")

    # Select ColorFix for ddcolor/stabilizer
    if ColorFix is None:
        ColorFix = "none"

    ColorFix = ColorFix.lower()
    color_fix = ['none', 'magenta', 'magenta/violet', 'violet', 'violet/red', 'blue/magenta', 'yellow', 'yellow/orange',
                 'yellow/green', 'retinex/red']
    hue_fix = ["none", "270:300", "250:360", "300:330", "300:360", "220:280", "60:90", "30:90", "60:120", "none"]

    co_id = 5
    try:
        co_id = color_fix.index(ColorFix)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: ColorFix choice is invalid for '" + ColorFix + "'")

    if tn_id == 0:
        hue_range = "none"
        hue_range2 = "none"
        # in this case all the Tweaks for DDcolor are disabled
        dd_tweak[0] = False
    elif tn_id != 0 and co_id == 0:
        hue_range = "none"
        hue_range2 = "none"
        # in this case the tweaks/denoise for DDcolor are enabled but hue adjust is disabled
        dd_tweak[0] = True
        dd_tweak[1] = True
    elif tn_id != 0 and co_id == 9:
        hue_range = hue_fix[4] +  "|" + hue_tune[2]
        hue_range2 = hue_fix[4] + "|" + hue_tune2[2]
        # in this case the tweaks/retinex for DDcolor are enabled and hue adjust is set to 'violet/red', tune = 'medium'
        dd_tweak[0] = True
        dd_tweak[2] = True
    else:
        hue_range = hue_fix[co_id] + "|" + hue_tune[tn_id]
        hue_range2 = hue_fix[co_id] + "|" + hue_tune2[tn_id]
        dd_tweak[0] = True  # in this case the Tweaks for DDcolor are enabled

    # Select Color Mapping
    ColorMap = ColorMap.lower()
    hue_w = ["1.0", "0.90", "0.80", "0.75"]
    colormap = ['none', 'blue->brown', 'blue->red', 'blue->green', 'green->brown', 'green->red', 'green->blue',
                'redrose->brown', 'redrose->blue', "red->brown", 'red->blue', 'yellow->rose']
    hue_map = ["none", "180:280|+140", "180:280|+100", "180:280|+220", "80:180|+260", "80:180|+220", "80:180|+140",
               "300:360,0:20|+40", "300:360,0:20|+260", "320:360|+50", "300:360|+260", "30:90|+300"]

    cl_id = 0
    try:
        cl_id = colormap.index(ColorMap)
    except ValueError:
        ret_range = restcolor._parse_hue_adjust(ColorMap)
        if ret_range is None:
            HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: ColorMap choice is invalid for '" + ColorMap + "'")
        else:
            cl_id = -1

    if cl_id == 0:
        chroma_adjust = "none"
        chroma_adjust2 = "none"
    elif cl_id == -1:
        chroma_adjust = ColorMap
        chroma_adjust2 = "none"
    else:
        chroma_adjust = hue_map[cl_id] + "," + hue_w[tn_id]
        if tn_id == 0:
            chroma_adjust2 = "none"
        else:
            chroma_adjust2 = chroma_adjust

    return dd_tweak, hue_range, hue_range2, chroma_adjust, chroma_adjust2

def _get_colormap(ColorMap: str = "red->brown", ColorTune: str ="light") -> str:

    color_tune = ['none', 'light', 'medium', 'strong']
    tn_id = 0
    try:
        tn_id = color_tune.index(ColorTune)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: ColorTune choice is invalid for '" + ColorTune + "'")

    # Select Color Mapping
    ColorMap = ColorMap.lower()
    hue_w = ["1.0", "0.90", "0.80", "0.75"]
    colormap = ['none', 'blue->brown', 'blue->red', 'blue->green', 'green->brown', 'green->red', 'green->blue',
                'redrose->brown', 'redrose->blue', "red->brown", 'red->blue', 'yellow->rose']
    hue_map = ["none", "180:280|+140", "180:280|+100", "180:280|+220", "80:180|+260", "80:180|+220", "80:180|+140",
               "300:360,0:20|+40", "300:360,0:20|+260", "320:360|+50", "300:360|+260", "30:90|+300"]

    cl_id = 0
    try:
        cl_id = colormap.index(ColorMap)
    except ValueError:
        ret_range = restcolor._parse_hue_adjust(ColorMap)
        if ret_range is None:
            HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: ColorMap choice is invalid for '" + ColorMap + "'")
        else:
            return ColorMap

    chroma_adjust = hue_map[cl_id] + "," + hue_w[tn_id]

    return chroma_adjust

def _get_tune_id(bw_tune: str) -> int:
    BWTune = bw_tune.lower()
    bw_tune_list = ['none', 'light', 'medium', 'strong']

    tn_id = bw_tune_list.index(BWTune)

    return tn_id

def _check_input(DeepExOnlyRefFrames: bool, ScFrameDir: str, DeepExMethod: int, ScThreshold: float,
                 ScMinFreq: int, DeepExRefMerge: int):
    if DeepExOnlyRefFrames and (ScFrameDir is None):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: DeepExOnlyRefFrames is enabled but ScFrameDir is unset")

    if not (ScFrameDir is None) and DeepExMethod != 0 and DeepExOnlyRefFrames:
        HAVC_LogMessage(MessageType.EXCEPTION,
                        "HAVC_main: DeepExOnlyRefFrames is enabled but method not = 0 (HAVC)")

    if (DeepExMethod != 0 and DeepExMethod != DEF_HAVC_METHOD_PLACEBO) and (ScFrameDir is None):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_main: DeepExMethod != 0 but ScFrameDir is unset")

    if DeepExMethod in (0, 1, 2, 5, 6, DEF_HAVC_METHOD_PLACEBO) and ScThreshold == 0 and ScMinFreq == 0:
        HAVC_LogMessage(MessageType.EXCEPTION,
                        "HAVC_main: DeepExMethod in (0, 1, 2, 5, 6) but ScThreshold and ScMinFreq are not set")

    if DeepExMethod in (2, 6) and DeepExRefMerge > 0:
        HAVC_LogMessage(MessageType.EXCEPTION,
                        "HAVC_main: RefMerge cannot be used with DeepExMethod in (2, 6)")


# ------------------------------------------------------------
# collection of small helper functions to validate parameters
# ------------------------------------------------------------

def is_limited_range(clip: vs.VideoNode) -> bool:
    # Try to read _ColorRange/_Range from props without forcing frame eval if possible.
    # Unfortunately, VapourSynth doesn't expose props without get_frame(),
    # so we have to accept minimal frame access—but make it safe.
    try:
        props = clip.get_frame(0).props
        if vs.core.core_version.release_major < 74:
            color_range = props.get('_ColorRange', vs.RANGE_LIMITED)  # default to limited if missing
        else:
            color_range = props.get('_Range', vs.RANGE_LIMITED)
        return color_range == vs.RANGE_LIMITED
    except Exception:
        # Fallback: assume full range if frame access fails
        return False

def _matrixIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    value = frame.props.get('_Matrix', None)

    # Non specificato o riservato
    if value in (None, 2, 3):
        return True

    # Non un membro valido dell'enum
    if value not in vs.MatrixCoefficients.__members__.values():
        return True

    # Coerenza con il color family
    if clip.format.color_family == vs.RGB and value != 0:
        return True  # RGB deve avere _Matrix=0
    if clip.format.color_family in (vs.YUV, vs.GRAY) and value == 0:
        return True  # YUV/GRAY non può avere _Matrix=RGB

    return False

def _transferIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    value = frame.props.get('_Transfer', None)
    return value in [None, 0, 2, 3] or value not in vs.TransferCharacteristics.__members__.values()


def _primariesIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    value = frame.props.get('_Primaries', None)
    return value in [None, 2] or value not in vs.ColorPrimaries.__members__.values()


def _rangeIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    if vs.core.core_version.release_major < 74:
        value = frame.props.get('_ColorRange', None)
        return value is None or value not in vs.ColorRange.__members__.values()
    else:
        value = frame.props.get('_Range', None)
        return value is None or value not in vs.Range.__members__.values()

def _fieldBaseIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    value = frame.props.get('_FieldBased', None)
    return value is None or value not in vs.FieldBased.__members__.values()


def adjust_rgb(clip: vs.VideoNode, factor: list = (1.0, 1.0, 1.0), bias: list = (0, 0, 0),
               gamma: list = (1.0, 1.0, 1.0)) -> vs.VideoNode:
    """Utility function to change the color and luminance of RGB clip.
       Gain, bias (offset) and gamma can be set independently on each channel.

       :param clip:         Clip to process. Only RGB24 format is supported.
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
    funcName = 'HAVC_adjust_rgb'

    rgb = clip

    # unpack rgb_factor
    r = factor[0]
    g = factor[1]
    b = factor[2]

    # unpack rgb_bias
    rb = bias[0]
    gb = bias[1]
    bb = bias[2]

    # unpack rgb_gamma
    rg = gamma[0]
    gg = gamma[1]
    bg = gamma[2]

    if rgb.format.color_family != vs.RGB:
        raise ValueError(funcName + ': input clip needs to be RGB!')

    rgb_type = rgb.format.sample_type
    size = 2 ** rgb.format.bits_per_sample
    # adjusting bias values rb,gb,bb for any RGB bit depth
    limited = is_limited_range(rgb)
    if limited:
        if rb > 235 or rb < -235:
            raise ValueError(funcName + ': source is flagged as "limited" but rb is out of range [-235,235]!')
        if gb > 235 or gb < -235:
            raise ValueError(funcName + ': source is flagged as "limited" but gb is out of range [-235,235]!')
        if bb > 235 or bb < -235:
            raise ValueError(funcName + ': source is flagged as "limited" but bb is out of range [-235,235]!')
    else:
        if rb > 255 or rb < -255:
            raise ValueError(funcName + ': source is flagged as "full" but rb is out of range [-255,255]!')
        if gb > 255 or gb < -255:
            raise ValueError(funcName + ': source is flagged as "full" but gb is out of range [-255,255]!')
        if bb > 255 or bb < -255:
            raise ValueError(funcName + ': source is flagged as "full" but bb is out of range [-255,255]!')

    if rg < 0:
        raise ValueError(funcName + ': rg needs to be >= 0!')
    if gg < 0:
        raise ValueError(funcName + ': gg needs to be >= 0!')
    if bg < 0:
        raise ValueError(funcName + ': bg needs to be >= 0!')

    if limited:
        if rgb_type == vs.INTEGER:
            maxVal = 235
        else:
            maxVal = 235.0
    else:
        if rgb_type == vs.INTEGER:
            maxVal = 255
        else:
            maxVal = 255.0
    rb, gb, bb = map(lambda b: b if size == maxVal else size / maxVal * b if rgb_type == vs.INTEGER else b / maxVal,
                     [rb, gb, bb])

    # x*r + rb , x*g + gb , x*b + bb
    rgb_adjusted = vs.core.std.Expr(rgb, [f"x {r} * {rb} +", f"x {g} * {gb} +", f"x {b} * {bb} +"])

    # gamma per channel
    planes = [vs.core.std.ShufflePlanes(rgb_adjusted, planes=p, colorfamily=vs.GRAY) for p in [0, 1, 2]]
    planes = [vs.core.std.Levels(planes[p], gamma=g) if not g == 1 else planes[p] for p, g in enumerate([rg, gg, bg])]
    rgb_adjusted = vs.core.std.ShufflePlanes(planes, planes=[0, 0, 0], colorfamily=vs.RGB)
    return rgb_adjusted


def rgb_denoise(clip: vs.VideoNode, denoise_levels: list[float] = (0.3, 0.2),
                 rgb_factors: list[float] = (0.98, 1.02, 1.0)) -> vs.VideoNode:

    w_strength = denoise_levels[0]
    b_strength = denoise_levels[1]

    r = rgb_factors[0]
    g = rgb_factors[1]
    b = rgb_factors[2]

    clip = clip.std.Levels(min_in=0, max_in=255, min_out=16, max_out=235)
    clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_in_s="full", range_s="limited")

    # step #1 : rgb colors are normalized and changed using rgb factors (this will change also the contrast/luminosity)
    clip = rgb_balance(clip=clip, strength=w_strength, rgb_factor=[r, g, b])
    # step #2 : the contrast/luminosity previously changed are adjusted/fixed using histogram equalization
    clip = rgb_equalizer(clip=clip, method=0, strength=b_strength, luma_blend=False, range_tv=True)

    clip = clip.std.Levels(min_in=16, max_in=235, min_out=0, max_out=255)
    clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_in_s="limited", range_s="full")

    return clip

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: histogram equalization (i.e. auto levels) implementation in python
             using OpenCV
URL: https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/
URL: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
------------------------------------------------------------------------------- 
"""
def vs_auto_levels(clip: vs.VideoNode = None, mode: str = 'Medium', method: int = 5,
                   luma_blend: bool = True, range_tv: bool = True) -> vs.VideoNode:
    """Pre-process filter using Retinex for improving contrast/luminosity of B&W clips
       to be colored with HAVC.

    :param clip:          Clip to process. Only RGB24 format is supported.
    :param mode:          This parameter allows to improve contrast and luminosity of input clip
                          Allowed values are:
                              'Light',
                              'Medium', (default)
                              'Strong'
    :param method:        Method used to perform the histogram equalization.
                         Allowed values are:
                            0 : Apply Contrast Limited Adaptive Histogram Equalization on Luma [41.5 fps] (default)
                            1 : Apply Simple Histogram Equalization on all RGB channels [54.5 fps]
                            2 : Apply CLAHE on all RGB channels [37.5 fps]
                            3 : method=0 and method=1 are merged [34.5]
                            4 : ScaleAbs + LUT [51.5 fps]
                            5 : Multi-Scale Retinex on Luma [45.5 fps]
    :param luma_blend:    If enabled the equalized image is blended with the original image, darker is the image and more
                          weight will be assigned to the original image. default = True
    :param range_tv:      If True, perform the color adjustments on limited TV range (the filter works better in TV range).
    """

    if not isinstance(clip, vs.VideoNode):
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_bw_tune: this is not a clip")

    bw_tune_p = mode.lower()
    bw_tune = ['none', 'light', 'medium', 'strong']
    b_strength = [0.0, 0.98, 0.99, 1.0]

    bw_id = 0
    try:
        bw_id = bw_tune.index(bw_tune_p)
    except ValueError:
        HAVC_LogMessage(MessageType.EXCEPTION, "HAVC_bw_tune: B&W tune choice is invalid: ", bw_tune_p)

    if range_tv:
        clip = clip.std.Levels(min_in=0, max_in=255, min_out=16, max_out=235)
        clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_in_s="full", range_s="limited")

    clip = rgb_equalizer(clip=clip, method=method, strength=b_strength[bw_id], luma_blend=luma_blend,
                         range_tv=range_tv)

    if range_tv:
        clip = clip.std.Levels(min_in=16, max_in=235, min_out=0, max_out=255)
        clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_in_s="limited", range_s="full")

    return clip


def rgb_equalizer(clip: vs.VideoNode, method: int = 0, clip_limit : float = 1.0, gridsize: int = 8,
                  strength: float = 0.5, weight3: float = 0.3, luma_blend: bool = True,
                  range_tv: bool = True) -> vs.VideoNode:
   """Histogram equalization implementation using OpenCV

   :param clip:          Clip to process (support only RGB24).
   :param method:        Method used to perform the histogram equalization.
                         Allowed values are:
                            0 : Apply Contrast Limited Adaptive Histogram Equalization on Luma [41.5 fps] (default)
                            1 : Apply Simple Histogram Equalization on all RGB channels [54.5 fps]
                            2 : Apply CLAHE on all RGB channels [37.5 fps]
                            3 : method=0 and method=1 are merged [34.5] 
                            4 : ScaleAbs + LUT [51.5 fps]
                            5 : Multi-Scale Retinex on Luma [45.5 fps]
   :param clip_limit:    Threshold for contrast limiting, range [0, 50] (default=1.0)
   :param gridsize:      Size of grid for histogram equalization. The input image will be divided into equally
                         sized rectangular tiles. gridsize defines the number of tiles in row and column.
                         Used by models: 0, 2, 3 (default=8)
   :param strength:      Strength of the filter. A strength=0 means that the clip is returned unchanged,
                         range [0, 1] (default=0.5)
   :param weight3:       Weight for method 3 (default=0.3)
   :param luma_blend:    If enabled the equalized image is blended with the original image, darker is the image and more
                         weight will be assigned to the original image. default = True
   :param range_tv:      If True, the clip is using limited TV range.
   """

   rgb_clip = clip
   rgb_orig = rgb_clip

   # A zero weight means that the clip filtered is returned unchanged and 1 means that original clip is returned
   weight: float = min(max(1.0 - strength, 0.0), 1.0)

   #  Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on Luma, to improve the contrast of images
   def frame_autolevels_CLAHE_yuv(n, f, limit : float = 2.0, gridsize: int = 8, blend: bool = True):

      img = frame_to_image(f)
      img_np = np.asarray(img)

      yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)

      y_image, u_image, v_image = cv2.split(yuv)

      if range_tv:
        maxrange = 235
        minrange = 16
        f_luma = max(round(np.mean(y_image) / maxrange, 6) - 0.07, 0)
      else:
        maxrange = 255
        minrange = 0
        f_luma = round(np.mean(y_image) / maxrange, 6)

      f_luma_bright = DEF_THT_DARK_BLACK <= f_luma <= DEF_THT_BRIGHT_WHITE

      if not f_luma_bright:
        #HAVC_LogMessage(MessageType.WARNING, "HAVC_bw_tune: frame ", n, " luma: ", f_luma)
        return f.copy()

      clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(gridsize, gridsize))
      y_image_eq = clahe.apply(y_image)
      yuv[:, :, 0] = y_image_eq.clip(min=minrange, max=maxrange).astype(int)
      img_new = Image.fromarray(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB))

      """
      if f_luma < 0.35:
          bright_scale = min(max(pow(f_luma / 0.35, 3.0), 0), 1)
          w = round(max(0.90 * bright_scale, 0.15), 5)
          HAVC_LogMessage(MessageType.WARNING, "HAVC_bw_tune: frame ", n, " luma: ", f_luma, " weight: ", w)
      else:
          HAVC_LogMessage(MessageType.WARNING, "HAVC_bw_tune: frame ", n, " luma: ", f_luma)
      """

      img_m = image_luma_blend(img, img_new, f_luma, 0.40, 0.90, 0.35, 2.0) if blend else img_new

      return image_to_frame(img_m, f.copy())

   # CLAHE (Contrast Limited Adaptive Histogram Equalization) on RGB
   def frame_autolevels_CLAHE_rgb(n, f, limit : float = 2.0, gridsize: int = 8, algo = 0, blend: bool = True):

      img = frame_to_image(f)

      if range_tv:
        f_luma = max(get_image_luma(img, 235) - 0.07, 0)
      else:
        f_luma = get_image_luma(img, 255)

      f_luma_bright = DEF_THT_DARK_BLACK <= f_luma <= DEF_THT_BRIGHT_WHITE

      if not f_luma_bright:
          return f.copy()

      img_np = np.asarray(img)
      r_image, g_image, b_image = cv2.split(img_np)

      if algo == 0:
         clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(gridsize, gridsize))

         r_image_eq = clahe.apply(r_image)
         g_image_eq = clahe.apply(g_image)
         b_image_eq = clahe.apply(b_image)
      else:
         r_image_eq = cv2.equalizeHist(r_image)
         g_image_eq = cv2.equalizeHist(g_image)
         b_image_eq = cv2.equalizeHist(b_image)

      img_new = Image.fromarray(cv2.merge((r_image_eq, g_image_eq, b_image_eq)))

      """
      if f_luma < 0.40:
          bright_scale = min(max(pow(f_luma / 0.40, 4.0), 0), 1)
          w = round(max(0.90 * bright_scale, 0.10), 5)
          HAVC_LogMessage(MessageType.WARNING, "HAVC_bw_tune: frame ", n, " luma: ", f_luma, " weight: ", w)
      else:
          HAVC_LogMessage(MessageType.WARNING, "HAVC_bw_tune: frame ", n, " luma: ", f_luma)
      """

      img_m = image_luma_blend(img, img_new, f_luma, 0.40, 0.90, 0.15, 4.0) if blend else img_new

      return image_to_frame(img_m, f.copy())

   # Automatic brightness and contrast optimization with optional histogram clipping
   # https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
   def autolevels_with_Scale(n, f, clip_hist_percent: float = 1.0, algo: int = 0, blend: bool = True):

      img = frame_to_image(f)

      if range_tv:
          maxrange = 235
          minrange = 16
          f_luma = max(get_image_luma(img, maxrange) - 0.07, 0)
      else:
          maxrange = 255
          minrange = 0
          f_luma = get_image_luma(img, maxrange)

      f_luma_bright = DEF_THT_DARK_BLACK <= f_luma <= DEF_THT_BRIGHT_WHITE

      if not f_luma_bright:
          return f.copy()

      img_np = np.asarray(img)
      gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

      # Calculate grayscale histogram
      hist = cv2.calcHist([gray], [0], None, [maxrange+1], [0, maxrange+1])
      hist_size = len(hist)

      # Calculate cumulative distribution from the histogram
      accumulator = [float(hist[0])]
      for index in range(1, hist_size):
         accumulator.append(accumulator[index - 1] + float(hist[index]))

      # Locate points to clip
      maximum = accumulator[-1]
      clip_hist_percent *= (maximum / 100.0)
      clip_hist_percent /= 2.0

      # Locate left cut
      minimum_gray = 0
      while accumulator[minimum_gray] < clip_hist_percent:
         minimum_gray += 1

      # Locate right cut
      maximum_gray = hist_size - 1
      while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
         maximum_gray -= 1

      # Calculate alpha and beta values
      alpha = maxrange / max(maximum_gray - minimum_gray, 1)
      beta = -minimum_gray * alpha

      if algo == 0:
         img_np_new = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
      else:
         # Add bias and gain to an image with saturation arithmetics. Unlike cv2.convertScaleAbs,
         # it does not take an absolute value, which would lead to nonsensical results
         # e.g., a pixel at 44 with alpha = 3 and beta = -210 becomes 78 with OpenCV, when in fact it should become 0.
         img_np_new = (img_np * alpha + beta).clip(min=minrange, max=maxrange).astype(np.uint8)

      img_new = Image.fromarray(img_np_new)

      """
      if f_luma < 0.35:
          bright_scale = min(max(pow(f_luma / 0.35, 3.0), 0), 1)
          w = round(max(0.90 * bright_scale, 0.15), 5)
          HAVC_LogMessage(MessageType.WARNING, "HAVC_bw_tune: frame ", n, " luma: ", f_luma, " weight: ", w)
      else:
          HAVC_LogMessage(MessageType.WARNING, "HAVC_bw_tune: frame ", n, " luma: ", f_luma)
      """

      img_m = image_luma_blend(img, img_new, f_luma, 0.40, 0.90, 0.25, 3.0) if blend else img_new

      return image_to_frame(img_m, f.copy())

   def ScaleAbs_LUT(clip: vs.VideoNode, lut_tune: float) -> vs.VideoNode:
       if lut_tune == 3:  # strong
           clip_a = vs_timecube(clip, strength=0.5, lut_effect=DEF_LUT_Amber_Light)
       elif lut_tune == 2:  # medium
           clip_a = vs_timecube(clip, strength=0.7, lut_effect=DEF_LUT_City_Skyline)
       else:  # light
           clip_a = vs_timecube(clip, strength=0.9, lut_effect=DEF_LUT_Exploration)

       return clip_a

   if method == 0:
      clip_a = rgb_clip.std.ModifyFrame(clips=rgb_clip, selector=partial(frame_autolevels_CLAHE_yuv, limit=clip_limit,
                                                                     gridsize=gridsize, blend=luma_blend))
   elif method == 1:
      clip_a = rgb_clip.std.ModifyFrame(clips=rgb_clip, selector=partial(frame_autolevels_CLAHE_rgb, algo=1, limit=clip_limit,
                                                                     gridsize=gridsize, blend=luma_blend))
   elif method == 2:
      clip_a = rgb_clip.std.ModifyFrame(clips=rgb_clip, selector=partial(frame_autolevels_CLAHE_rgb, algo=0, limit=clip_limit,
                                                                     gridsize=gridsize, blend=luma_blend))
   elif method == 3:
      clip_a = rgb_clip.std.ModifyFrame(clips=rgb_clip, selector=partial(frame_autolevels_CLAHE_yuv, limit=clip_limit,
                                                                       gridsize=gridsize, blend=luma_blend))
      clip_b = rgb_clip.std.ModifyFrame(clips=rgb_clip, selector=partial(frame_autolevels_CLAHE_rgb, algo=1, limit=clip_limit,
                                                                      gridsize=gridsize, blend=luma_blend))
      # weight=0 means that is returned clip_a, weight=1 means that is returned clip_b
      clip_a = vs.core.std.Merge(clip_a, clip_b, weight3)
   elif method == 4:
       clip_a = ScaleAbs_LUT(rgb_clip, weight3)
       # clip_a = rgb_clip.std.ModifyFrame(clips=clip_a, selector=partial(autolevels_with_Scale, algo=0, blend=luma_blend,
       #                                                  clip_hist_percent=clip_limit))
   else:
      clip_a = vs_retinex(rgb_clip, luma_dark=0.20, luma_bright=0.80, sigmas=[25, 80, 250], range_tv_in=range_tv,
                          range_tv_out=range_tv, blend=luma_blend)

   # A zero weight means that clip_a is returned unchanged and 1 means that rgb_orig is returned unchanged
   if 0 <= weight < 1:
      clip_rgb = vs.core.std.Merge(clip_a, rgb_orig, weight)
   else:
      clip_rgb = rgb_orig  # is returned the original clip

   if clip.format.id != vs.RGB24:
      # convert the format for tweak to YUV 8bits
      clip_new = clip_rgb.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="limited")
   else:
      clip_new = clip_rgb

   return clip_new

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: function that takes a video clip as an input and calculates the average color values 
for each of the three color planes (red, green, blue), with optional RGB scaling factors.
The white balance of the input clip is adjusted based on the color balance of the individual frames.
URL: http://www.vapoursynth.com/doc/functions/frameeval.html
------------------------------------------------------------------------------- 
"""
def rgb_balance(clip: vs.VideoNode, strength: float = 0.5, rgb_factor: list = (1.0, 1.0, 1.0) ) -> vs.VideoNode:
   """ Auto white balance filter using PlaneStats()

    :param clip:           Clip to process (support only RGB24).
    :param strength:       Strength of the filter. A strength=0 means that the clip is returned unchanged,
                           range [0, 1] (default=0.5)
    :param rgb_factor:     List of Red, Green and Blue scaling factor, in the list format: (r, g, b),
                           default = (1, 1, 1). For example, r=1.3 multiplies the red channel pixel values by 1.3.

   """
   rgb_clip = clip

   # A zero weight means that the clip filtered is returned unchanged and 1 means that original clip is returned
   weight: float = min(max(1.0 - strength, 0.0), 1.0)

   # auto white from http://www.vapoursynth.com/doc/functions/frameeval.html
   def frame_autowhite(n, f, clip, core, rgb_fact):
      small_number = 0.000000001
      # unpack rgb_factor
      r = rgb_fact[0]
      g = rgb_fact[1]
      b = rgb_fact[2]
      red = f[0].props['PlaneStatsAverage']
      green = f[1].props['PlaneStatsAverage']
      blue = f[2].props['PlaneStatsAverage']
      max_rgb = max(red, green, blue)
      red_corr = max_rgb / max(red, small_number)
      green_corr = max_rgb / max(green, small_number)
      blue_corr = max_rgb / max(blue, small_number)
      norm = max(blue, math.sqrt(red_corr * red_corr + green_corr * green_corr + blue_corr * blue_corr) / math.sqrt(3),
                 small_number)
      r_gain = round(r * red_corr / norm, 8)
      g_gain = round(g * green_corr / norm, 8)
      b_gain = round(b * blue_corr / norm, 8)
      return core.std.Expr(clip,
                           expr=['x ' + repr(r_gain) + ' *', 'x ' + repr(g_gain) + ' *', 'x ' + repr(b_gain) + ' *'])

   r_avg = vs.core.std.PlaneStats(rgb_clip, plane=0)
   g_avg = vs.core.std.PlaneStats(rgb_clip, plane=1)
   b_avg = vs.core.std.PlaneStats(rgb_clip, plane=2)

   clip_a = vs.core.std.FrameEval(rgb_clip, partial(frame_autowhite, clip=rgb_clip, core=vs.core,
                                                           rgb_fact=rgb_factor), prop_src=[r_avg, g_avg, b_avg])

   clip_b = rgb_clip

   # A zero weight means that clip_a is returned unchanged and 1 means that clip_b is returned unchanged
   if 0 <= weight < 1:
      clip_rgb = vs.core.std.Merge(clip_a, clip_b, weight)
   else:
      clip_rgb = rgb_clip  # is returned the original clip

   if clip.format.id != vs.RGB24:
      # convert the format for tweak to YUV 8bits
      clip_new = clip_rgb.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="limited")
   else:
      clip_new = clip_rgb

   return clip_new
