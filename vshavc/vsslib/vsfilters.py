"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2025-10-26
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Library of Vapoursynth filter functions.
"""
import vapoursynth as vs
import os
import math
from functools import partial

from vsdeoldify.vsslib.imfilters import _color_temporal_stabilizer, _chroma_temporal_limiter, get_image_luma
from vsdeoldify.vsslib.imfilters import image_chroma_tweak, image_luma_merge, w_image_luma_merge, image_tweak
from vsdeoldify.vsslib.imfilters import luma_adjusted_levels, chroma_post_process
from vsdeoldify.vsslib.restcolor import restore_color, restore_color_gradient, adjust_hue_range
from vsdeoldify.vsslib.vsutils import frame_to_image, image_to_frame

from vsdeoldify.vsslib.constants import *

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
Function which try to stabilize the colors of a clip using color temporal stabilizer.
As stabilizer is used the Vapoursynth function "std.AverageFrames()", the mode, can
be: "arithmetic", "weighted"
"""


def vs_clip_color_stabilizer(clip: vs.VideoNode = None, nframes: int = 5, mode: str = "A",
                             scenechange: bool = True) -> vs.VideoNode:
    """Stabilise clip colours using std.AverageFrames on the U and V chroma planes.

    Converts to YUV420P8, applies AverageFrames, and converts back to RGB24.

    :param clip:        RGB24 input clip.
    :param nframes:     Number of frames to average (forced odd). Range [3, 15]. Default 5.
    :param mode:        Averaging mode: 'A' = arithmetic, 'W' = weighted (centre-biased).
    :param scenechange: If True, resets the temporal average at scene changes.
    :return:            Chroma-stabilised RGB24 clip.
    """
    if nframes % 2 == 0:
        nframes += 1

    N = max(3, min(nframes, 15))

    match mode:
        case "A" | "arithmetic" | "center":  # for compatibility with version 2.0.0
            weight_list = _build_avg_arithmetic(N)
        case "W" | "weighted" | "left" | "right":  # for compatibility with version 2.0.0
            weight_list = _build_avg_weighted(N)
        case _:
            raise vs.Error("HybridAVC: unknown average method: " + mode)

            # vs.core.log_message(2, "weight_list= " + str(len(weight_list)))

    # convert the clip format for AverageFrames to YUV    
    clip_yuv = clip.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")
    # apply AverageFrames to YUV colorspace      
    clip_yuv = vs.core.std.AverageFrames(clip_yuv, weight_list, scale=100, scenechange=scenechange, planes=[1, 2])
    # convert the clip format for deoldify to RGB24 
    clip_rgb = clip_yuv.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full",
                                       dither_type="error_diffusion")

    return clip_rgb


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
Filter which try to stabilize the colors of a clip using color temporal stabilizer.
The stabilization is performed by averaging past/future frames. Since the non matched
areas of past/future frames are gray because is missing in the past/future the color 
information, the filter will fill the gray areas with the pixels of current frames 
(eventually de-saturated with the parameter "sat"). The image restored in this way is
blended with the non restored image using the parameter "weight" (if =0 no blending 
is applied). The gray areas are selected by the threshold parameter "tht". All the pixels
in the HSV color space with "S" < "tht" will be considered gray. If "tht=0" no color
frame restore is applied.
"""


def vs_chroma_stabilizer_ex(clip: vs.VideoNode = None, nframes: int = 5, mode: str = "A", sat: float = 1.0,
                            tht: int = 0, weight: float = 0.5, tht_scen: float = 0.8, hue_adjust: str = 'none',
                            algo: int = 0) -> vs.VideoNode:
    """Extended chroma stabiliser with optional gray-pixel colour restoration.

    When tht==0 delegates to vs_clip_color_stabilizer (no restoration). When tht>0,
    restores gray pixels from neighbouring frames using restore_color before averaging.

    :param clip:       RGB24 input clip.
    :param nframes:    Number of frames for temporal averaging (forced odd). Range [3, 15]. Default 5.
    :param mode:       Averaging mode: 'A' = arithmetic, 'W' = weighted.
    :param sat:        Saturation applied to restored gray pixels. Default 1.0.
    :param tht:        Saturation threshold to detect gray pixels [0, 255]. 0 = no restoration.
    :param weight:     Blend weight between restored and original frame. Default 0.5.
    :param tht_scen:   Scene-change detection threshold for restoration. Default 0.8.
    :param hue_adjust: Chroma adjustment string applied to the final frame. 'none' = disabled.
    :param algo:       Stabiliser algorithm: 0 = AverageFrames-based, 1 = ModifyFrame-based.
    :return:           Chroma-stabilised RGB24 clip.
    """
    if tht == 0:
        return vs_clip_color_stabilizer(clip, nframes, mode, scenechange=True)

    if nframes % 2 == 0:
        nframes += 1

    N = max(3, min(nframes, 15))

    match mode:
        case "A" | "arithmetic" | "center":  # for compatibility with version 2.0.0
            weight_list = _build_avg_arithmetic(N)
        case "W" | "weighted" | "left" | "right":  # for compatibility with version 2.0.0
            weight_list = _build_avg_weighted(N)
        case _:
            raise vs.Error("HybridAVC: unknown average method: " + mode)

            # vs.core.log_message(2, "algo= " + str(algo))

    if algo == 0:
        clip_rgb = _average_clips_ex(clip=clip, weight_list=weight_list, sat=sat, tht=tht, weight=weight,
                                     tht_scen=tht_scen, hue_adjust="none")
    else:
        clip_rgb = _average_frames_ex(clip=clip, weight_list=weight_list, sat=sat, tht=tht, weight=weight,
                                      tht_scen=tht_scen, hue_adjust="none")

    # hue adjustment applied only on the final frame
    clip_rgb = vs_adjust_clip_hue(clip=clip_rgb, hue_adjust=hue_adjust)

    return clip_rgb


def _build_avg_arithmetic(nframes: int = 5) -> list:
    """Build an arithmetic (equal-weight) integer weight list for std.AverageFrames.

    Weights sum to 100. The centre frame receives any rounding remainder.

    :param nframes: Total number of frames (must be odd). Default 5.
    :return:        List of integer weights summing to 100.
    """
    N = nframes
    Nh = round((N - 1) / 2)
    Wi = math.trunc(100.0 / N)

    Wc = 100 - (N - 1) * Wi

    weight_list = list()

    for i in range(0, Nh):
        weight_list.append(Wi)
    weight_list.append(Wc)
    for i in range(0, Nh):
        weight_list.append(Wi)

    return weight_list


def _build_avg_weighted(nframes: int = 5) -> list:
    """Build a centre-weighted integer weight list for std.AverageFrames.

    The centre frame receives twice the weight of flanking frames; weights sum to 100.

    :param nframes: Total number of frames (must be odd). Default 5.
    :return:        List of integer weights summing to 100.
    """
    N = nframes
    Nh = round((N - 1) / 2)

    WBase = N * (N + 1) * 0.5

    Wi_scale = 1
    Wc_scale = 2

    SumWi = 0
    weight_list = list()
    for i in range(0, Nh):
        Wi = math.trunc(Wi_scale * 100 * (i + 1) / WBase)
        SumWi += Wi
        weight_list.append(Wi)
    Wc = 100 - Wc_scale * SumWi
    weight_list.append(Wc)
    for i in range(0, Nh):
        Wi = math.trunc(Wi_scale * 100 * (i + 1) / WBase)
        weight_list.append(Wi)

    return weight_list


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function which try to stabilize the colors of a clip using color temporal stabilizer,
the colors of current frame will be averaged with the ones of previous frames.  
(based on ModifyFrame)
"""


def _average_frames_ex(clip: vs.VideoNode = None, weight_list: list = None, sat: float = 1.0, tht: int = 0,
                       weight: float = 0.2, tht_scen: float = 0.8, hue_adjust: str = 'none') -> vs.VideoNode:
    """Temporal colour stabiliser using ModifyFrame: averages chroma across past and future frames.

    For each frame, neighbouring frames are fetched via clip.get_frame(), their gray pixels
    are restored using restore_color, and the results are averaged with _color_temporal_stabilizer.

    :param clip:        RGB24 input clip.
    :param weight_list: Per-frame integer weights (centre frame at index Nh). Sums to 100.
    :param sat:         Saturation for restored gray pixels. Default 1.0.
    :param tht:         Saturation threshold to detect gray pixels. Default 0.
    :param weight:      Blend weight between restored and original frame. Default 0.2.
    :param tht_scen:    Scene-change ratio threshold for restore_color. Default 0.8.
    :param hue_adjust:  Chroma adjustment string. 'none' = disabled.
    :return:            Temporally stabilised RGB24 clip.
    """
    def smooth_frame(n, f, clip_base: vs.VideoNode = None, weight_list: list = None, sat: float = 1.0, tht: int = 0,
                     weight: float = 0.2, tht_scen: float = 0.8, hue_adjust: str = 'none') -> vs.VideoFrame:
        max_frames = len(weight_list)
        tot_frames = clip_base.num_frames - 10
        f_out = f.copy()
        if n < max_frames or n > tot_frames:
            return f_out
        img_f = list()
        img_base = frame_to_image(f)
        Nh = round((max_frames - 1) / 2)
        for i in range(0, Nh):
            Ni = n - (Nh - i)
            img_f.append(
                restore_color(img_base, frame_to_image(clip_base.get_frame(Ni)), sat, tht, weight, tht_scen, hue_adjust,
                              False))
        img_f.append(img_base)
        for i in range(0, Nh):
            Ni = n + (i + 1)
            img_f.append(
                restore_color(img_base, frame_to_image(clip_base.get_frame(Ni)), sat, tht, weight, tht_scen, hue_adjust,
                              False))
        img_m = _color_temporal_stabilizer(img_f, weight_list)
        return image_to_frame(img_m, f_out)

    clip = clip.std.ModifyFrame(clips=[clip],
                                selector=partial(smooth_frame, clip_base=clip, weight_list=weight_list, sat=sat,
                                                 tht=tht, weight=weight, tht_scen=tht_scen, hue_adjust=hue_adjust))
    return clip


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function which try to stabilize the colors of a clip using color temporal stabilizer,
the colors of current frame will be averaged with the ones of previous frames.
(based on AverageFrames)
"""


def _average_clips_ex(clip: vs.VideoNode = None, weight_list: list = None, sat: float = 1.0, tht: int = 0,
                      weight: float = 0.2, tht_scen: float = 0.8, hue_adjust: str = 'none') -> vs.VideoNode:
    """Temporal colour stabiliser using AverageFrames: averages chroma across shifted clips.

    Builds offset clips via vs_get_clip_frame and vs_recover_clip_color, converts them
    to YUV420P8, and averages with std.AverageFrames on chroma planes.

    :param clip:        RGB24 input clip.
    :param weight_list: Per-frame integer weights. Sums to 100.
    :param sat:         Saturation for restored gray pixels. Default 1.0.
    :param tht:         Saturation threshold to detect gray pixels. Default 0.
    :param weight:      Blend weight between restored and original frame. Default 0.2.
    :param tht_scen:    Scene-change ratio threshold for restore_color. Default 0.8.
    :param hue_adjust:  Chroma adjustment string. 'none' = disabled.
    :return:            Temporally stabilised RGB24 clip.
    """
    max_frames = len(weight_list)
    clips: list[vs.VideoNode] = list()
    clip_yuv = clip.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full", dither_type="error_diffusion")
    Nh = round((max_frames - 1) / 2)
    for i in range(0, Nh):
        Ni = -(Nh - i)
        clip_i = vs_get_clip_frame(clip=clip, nframe=Ni)
        clip_i = vs_recover_clip_color(clip=clip_i, clip_color=clip, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen,
                                       hue_adjust=hue_adjust, return_mask=False)
        clips.append(
            clip_i.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full", dither_type="error_diffusion"))
    clips.append(clip_yuv)
    for i in range(0, Nh):
        Ni = i + 1
        clip_i = vs_get_clip_frame(clip=clip, nframe=Ni)
        clip_i = vs_recover_clip_color(clip=clip_i, clip_color=clip, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen,
                                       hue_adjust=hue_adjust, return_mask=False)
        clips.append(
            clip_i.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full", dither_type="error_diffusion"))
    clip_avg = vs.core.std.AverageFrames(clips=clips, weights=weight_list, scale=100, planes=[1, 2])
    # convert the clip format for deoldify to RGB24 
    clip_rgb = clip_avg.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full",
                                       dither_type="error_diffusion")

    return clip_rgb


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to function AverageFrames() to get frames fast.
"""


def vs_get_clip_frame(clip: vs.VideoNode, nframe: int = 0) -> vs.VideoNode:
    """Return a clip where every frame is a copy of the frame at offset nframe.

    Uses std.AverageFrames with a weight of 100 on the target offset and 0 on all others
    to cheaply extract a frame at a relative position without looping. Positive nframe
    fetches a future frame; negative fetches a past frame.

    :param clip:   Input clip (any format).
    :param nframe: Frame offset in [-15, +15]. 0 = return clip unchanged.
    :return:       Clip where every output frame is taken from position n+nframe.
    """
    if nframe == 0:
        return clip

    n = abs(nframe)

    if n > 15:
        raise vs.Error("HybridAVC: nframe must be between: -15, +15")

    weights_list = list()

    for i in range(-n, n + 1):
        if i == nframe:
            weights_list.append(100)
        else:
            weights_list.append(0)

    vs_format = clip.format.id

    # clip converted
    clip_yuv = clip.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")

    # apply AverageFrames to YUV colorspace      
    clip_yuv = vs.core.std.AverageFrames(clip_yuv, weights_list, scale=100, scenechange=False, planes=[1, 2])

    # convert to the original clip format
    if clip.format.color_family == "YUV":
        clip = clip_yuv.resize.Bicubic(format=vs_format)
    else:
        clip = clip_yuv.resize.Bicubic(format=vs_format, matrix_in_s="709", range_s="full",
                                       dither_type="error_diffusion")

    return clip


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function restore_color() to restore gray frames.
"""


def vs_recover_clip_color(clip: vs.VideoNode = None, clip_color: vs.VideoNode = None, sat: float = 1.0, tht: int = 0,
                          weight: float = 0.2, tht_scen: float = 0.8, hue_adjust: str = 'none',
                          return_mask: bool = False) -> vs.VideoNode:
    """Restore gray pixel colours in clip from clip_color (all frames, no scene-change check).

    Wrapper around vs_sc_recover_clip_color with scenechange=False.

    :param clip:        Target clip with gray areas to restore (RGB24).
    :param clip_color:  Source clip providing replacement colours (RGB24).
    :param sat:         Saturation multiplier for clip_color. Default 1.0.
    :param tht:         Saturation threshold for gray-pixel detection [0, 255]. Default 0.
    :param weight:      Post-merge blend weight. >0 blends toward original, <0 toward clip_color.
    :param tht_scen:    Gray-frame ratio threshold; above it the original frame is returned.
    :param hue_adjust:  Chroma adjustment string. 'none' = disabled.
    :param return_mask: If True, return the gray-pixel mask.
    :return:            Colour-restored RGB24 clip (or mask clip).
    """
    return vs_sc_recover_clip_color(clip, clip_color, sat, tht, weight, tht_scen, hue_adjust, return_mask, scenechange=False)

def vs_sc_recover_clip_color(clip: vs.VideoNode = None, clip_color: vs.VideoNode = None, sat: float = 1.0, tht: int = 0,
                          weight: float = 0.2, tht_scen: float = 0.8, hue_adjust: str = 'none',
                          return_mask: bool = False, scenechange: bool = False) -> vs.VideoNode:
    """Utility function to recover the colors of gray pixels in clip by using the colors provided in clip_color by
       using a simple binary merge.

        Args:
            clip:          clip to repair the colors, only RGB24 format is supported
            clip_color:    clip with the colors to restore, only RGB24 format is supported
            sat:           this parameter allows to change the saturation of colored clip (default = 0.8)
            tht:           threshold to identify gray pixels, range[0, 255] (default = 30)
            weight:        represent the weight for merging the masked colored clip with clip_a/clip_b:
                                         if weight > 0: merge(clip_restored, clip, weight)
                                         if weight < 0: merge(clip_restored, clip_colored, weight)
                                     Range[0,1], default = 0 (is returned clip_restored without additional merge)
            hue_adjust:    parameter used to control the steepness of gradient curve, values above the default value
                           will preserve more pixels, but could introduce some artifacts, range[1, 10] (default = 2)
            return_mask:   if True, will be returned the mask used to identify the gray pixels (white region), could
                           be useful to visualize the gradient mask for debugging, (default = false).
            scenechange:   if True, the filter will be applied only on the scene-change frames, (default = false)
    """
    def color_frame(n, f, sat: float = 1.0, tht: int = 0, weight: float = 0.2, tht_scen: float = 0.8,
                    hue_adjust: str = 'none', return_mask: bool = False, scenechange: bool = False):

        if scenechange:
            is_scenechange = (n == 0) or (f[0].props.get('_SceneChangePrev', 0) == 1)
            if not is_scenechange:
                return f[0].copy()

        f_out = f[0].copy()

        if n < 15:
            return f_out

        img_gray = frame_to_image(f[0])
        img_color = frame_to_image(f[1])

        f_luma = get_image_luma(img_gray, 255)

        f_luma_standard = DEF_STANDARD_DARK <= f_luma <= DEF_STANDARD_BRIGHT

        if not f_luma_standard:
            weight = min(weight, -0.8)

        img_restored = restore_color(img_color, img_gray, sat, tht, weight, tht_scen, hue_adjust, return_mask)
        return image_to_frame(img_restored, f_out)

    clip = clip.std.ModifyFrame(clips=[clip, clip_color],
                                selector=partial(color_frame, sat=sat, tht=tht, weight=weight, tht_scen=tht_scen,
                                                 hue_adjust=hue_adjust, return_mask=return_mask, scenechange=scenechange))
    return clip


def vs_recover_gradient_color(clip: vs.VideoNode = None, clip_color: vs.VideoNode = None, sat: float = 1.0,
                              tht: int = 15, weight: float = 0.0, alpha: float = 2.0,
                              return_mask: bool = False) -> vs.VideoNode:
    """Restore gray pixel colours using a gradient mask (all frames, no scene-change check).

    Wrapper around vs_sc_recover_gradient_color with scenechange=False.

    :param clip:        Target clip with gray areas to restore (RGB24).
    :param clip_color:  Source clip providing replacement colours (RGB24).
    :param sat:         Saturation multiplier for clip_color. Default 1.0.
    :param tht:         Saturation threshold for gradient detection [0, 255]. Default 15.
    :param weight:      Post-merge blend weight. Default 0.0.
    :param alpha:       Gradient steepness. Default 2.0.
    :param return_mask: If True, return the gradient mask clip.
    :return:            Colour-restored RGB24 clip (or mask clip).
    """
    return vs_sc_recover_gradient_color(clip, clip_color, sat, tht, weight, alpha, return_mask, scenechange=False)


def vs_sc_recover_gradient_color(clip: vs.VideoNode = None, clip_color: vs.VideoNode = None, sat: float = 1.0,
                              tht: int = 15, weight: float = 0.0, alpha: float = 2.0,
                              return_mask: bool = False, scenechange: bool = False, algo: int = 0) -> vs.VideoNode:
    """Utility function to recover the colors of gray pixels in clip by using the colors provided in clip_color by
       using a gradient merge.

    Args:
        clip:          clip to repair the colors, only RGB24 format is supported
        clip_color:    clip with the colors to restore, only RGB24 format is supported
        sat:           this parameter allows to change the saturation of colored clip (default = 0.8)
        tht:           threshold to identify gray pixels, range[0, 255] (default = 30)
        weight:        represent the weight for merging the masked colored clip with clip_a/clip_b:
                                     if weight > 0: merge(clip_restored, clip, weight)
                                     if weight < 0: merge(clip_restored, clip_colored, weight)
                                 Range[0,1], default = 0 (is returned clip_restored without additional merge)
        alpha:         parameter used to control the steepness of gradient curve, values above the default value
                       will preserve more pixels, but could introduce some artifacts, range[1, 10] (default = 2)
        return_mask:   if True, will be returned the mask used to identify the gray pixels (white region), could
                       be useful to visualize the gradient mask for debugging, (default = false).
        scenechange:   if True, the filter will be applied only on the scene-change frames, (default = false)
        algo:          algorithm to build the mask, allowed values are:
                            [0] = Linear decay with steep gradient, (default)
                            [1] = Linear decay
                            [2] = Exponential decay
       """
    def color_grad_frame(n, f, sat: float, tht: int, weight: float, alpha: float, return_mask: bool,
                         scenechange: bool, algo: int):

        if scenechange:
            is_scenechange = (n == 0) or (f[0].props.get('_SceneChangePrev', 0) == 1)
            if not is_scenechange:
                return f[0].copy()

        f_out = f[0].copy()
        img_gray = frame_to_image(f[0])
        img_color = frame_to_image(f[1])

        f_luma = get_image_luma(img_gray, 255)

        f_luma_standard = DEF_STANDARD_DARK <= f_luma <= DEF_STANDARD_BRIGHT

        if not f_luma_standard:
            weight = min(weight, -0.5)
            alpha = max(alpha, 4.0)

        img_restored = restore_color_gradient(img_color, img_gray, sat, tht, weight, alpha, return_mask, algo)
        return image_to_frame(img_restored, f_out)

    clip = clip.std.ModifyFrame(clips=[clip, clip_color],
                                selector=partial(color_grad_frame, sat=sat, tht=tht, weight=weight, alpha=alpha,
                                                 return_mask=return_mask, scenechange=scenechange, algo=algo))

    #clip = debug_ModifyFrame(f_start=33162, f_end=33175, clip=clip, clips=[clip, clip_color],
    #                         selector=partial(color_grad_frame, sat=sat, tht=tht, weight=weight, alpha=alpha,
    #                                          return_mask=return_mask))

    return clip


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function restore_color() to restore gray frames.
"""


def vs_sc_adjust_clip_hue(clip: vs.VideoNode = None, hue_adjust: str = 'none',
                          scenechange: bool = True) -> vs.VideoNode:
    """Apply a targeted hue-range colour adjustment to a clip with optional scene-change gating.

    :param clip:        RGB24 input clip.
    :param hue_adjust:  Chroma adjustment string (e.g. "300:360|0.8,0.1"). 'none' = bypass.
    :param scenechange: If True, process only scene-change frames. Default True.
    :return:            Hue-adjusted RGB24 clip.
    """
    if hue_adjust == "" or hue_adjust == "none":
        return clip

    def color_frame(n, f, hue_adjust: str = 'none', scenechange: bool = True):

        if scenechange:
            is_scenechange = (n == 0) or (f.props.get('_SceneChangePrev', 0) == 1)
            if not is_scenechange:
                return f.copy()

        img_color = frame_to_image(f)
        img_restored = adjust_hue_range(img_color, hue_adjust=hue_adjust)

        return image_to_frame(img_restored, f.copy())

    clip = clip.std.ModifyFrame(clips=clip,
                                selector=partial(color_frame, hue_adjust=hue_adjust, scenechange=scenechange))

    return clip


def vs_adjust_clip_hue(clip: vs.VideoNode = None, hue_adjust: str = 'none') -> vs.VideoNode:
    """Apply a targeted hue-range colour adjustment to all frames (no scene-change gating).

    Wrapper around vs_sc_adjust_clip_hue with scenechange=False.

    :param clip:       RGB24 input clip.
    :param hue_adjust: Chroma adjustment string. 'none' = bypass.
    :return:           Hue-adjusted RGB24 clip.
    """
    return vs_sc_adjust_clip_hue(clip, hue_adjust, False)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
Function to which try to stabilize the chroma of a clip using chroma temporal limiter,
the chroma of current frame will be forced to be inside the range defined by max_deviation  
"""


def vs_chroma_limiter(clip: vs.VideoNode = None, deviation: float = 0.05) -> vs.VideoNode:
    """Limit per-frame chroma (U, V) deviation to at most ±deviation from the previous frame.

    :param clip:      RGB24 input clip.
    :param deviation: Maximum fractional chroma change per frame. Clamped to [0.01, 0.50]. Default 0.05.
    :return:          Chroma-limited RGB24 clip.
    """
    max_deviation = max(min(deviation, 0.5), 0.01)

    def limit_chroma_frame(n, f, clip_base: vs.VideoNode = None, max_deviation: float = 0.05):
        f_out = f.copy()
        if n == 0:
            return f_out
        cur_img = frame_to_image(f)
        prv_img = frame_to_image(clip_base.get_frame(n - 1))
        img_m = _chroma_temporal_limiter(cur_img, prv_img, max_deviation)
        return image_to_frame(img_m, f_out)

    clip = clip.std.ModifyFrame(clips=[clip],
                                selector=partial(limit_chroma_frame, clip_base=clip, max_deviation=max_deviation))
    return clip


def _frame_chroma_stabilizer(clip: vs.VideoNode = None, max_deviation: float = 0.05) -> vs.VideoNode:
    """Limit per-frame chroma deviation via ModifyFrame (internal implementation).

    :param clip:          RGB24 input clip.
    :param max_deviation: Maximum fractional chroma deviation. Default 0.05.
    :return:              Chroma-limited RGB24 clip.
    """
    def limit_chroma_frame(n, f, clip_base: vs.VideoNode = None, max_deviation: float = 0.05):
        f_out = f.copy()
        if n == 0:
            return f_out
        cur_img = frame_to_image(f)
        prv_img = frame_to_image(clip_base.get_frame(n - 1))
        img_m = _chroma_temporal_limiter(cur_img, prv_img, max_deviation)
        return image_to_frame(img_m, f_out)

    clip = clip.std.ModifyFrame(clips=[clip],
                                selector=partial(limit_chroma_frame, clip_base=clip, max_deviation=max_deviation))
    return clip


def _clip_chroma_stabilizer(clip: vs.VideoNode = None, max_deviation: float = 0.05) -> vs.VideoNode:
    """Limit per-frame chroma deviation via FrameEval (alternative internal implementation).

    :param clip:          RGB24 input clip.
    :param max_deviation: Maximum fractional chroma deviation. Default 0.05.
    :return:              Chroma-limited RGB24 clip.
    """
    def limit_chroma_frame(n, f, clip_base: vs.VideoNode = None, max_deviation: float = 0.05):
        return _frame_chroma_stabilizer(clip_base, max_deviation)

    clip = clip.std.FrameEval(clip, eval=partial(limit_chroma_frame, clip_base=clip, max_deviation=max_deviation),
                              prop_src=[clip])
    return clip


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
The the pixels with luma below dark_threshold will be desaturared to level defined
by the dark_sat parameter.
"""


def vs_sc_chroma_bright_tweak(clip: vs.VideoNode = None, black_threshold: float = 0.3, white_threshold: float = 0.6,
                              dark_sat: float = 0.8, dark_bright: float = -0.10, scenechange: bool = True,
                              chroma_adjust: str = 'none') -> vs.VideoNode:
    """Desaturate and darken pixels whose luma is below black_threshold.

    A luma-based gradient mask blends the tweaked frame with the original between
    black_threshold and white_threshold. Optionally applies a direct colour mapping.

    :param clip:             RGB24 input clip.
    :param black_threshold:  Luma below which the tweak is fully applied (fraction [0, 1]).
    :param white_threshold:  Luma above which no tweak is applied; gradient in between.
    :param dark_sat:         Saturation multiplier for dark pixels. Default 0.8.
    :param dark_bright:      Brightness offset for dark pixels. Default -0.10.
    :param scenechange:      If True, process only scene-change frames. Default True.
    :param chroma_adjust:    Chroma adjustment string applied before luma merge. 'none' = disabled.
    :return:                 Tweaked RGB24 clip.
    """
    def merge_frame(n, f, black_limit: float = 0.3, white_limit: float = 0.6, dark_bright: float = -0.10,
                    dark_sat: float = 0.8, scenechange: bool = True, chroma_adjust: str = 'none'):

        if scenechange:
            is_scenechange = (n == 0) or (f.props.get('_SceneChangePrev', 0) == 1)
            if not is_scenechange:
                return f.copy()

        img1 = frame_to_image(f)
        img2 = image_chroma_tweak(img1, bright=dark_bright, sat=dark_sat, hue_adjust=chroma_adjust)
        if black_limit == white_limit:
            img_m = image_luma_merge(img2, img1, black_limit)
        else:
            img_m = w_image_luma_merge(img2, img1, black_limit, white_limit)
        return image_to_frame(img_m, f.copy())

    return clip.std.ModifyFrame(clips=clip,
                                selector=partial(merge_frame, black_limit=black_threshold, white_limit=white_threshold,
                                                 dark_bright=dark_bright, dark_sat=dark_sat, scenechange=scenechange,
                                                 chroma_adjust=chroma_adjust))


def vs_chroma_bright_tweak(clip: vs.VideoNode = None, black_threshold: float = 0.3, white_threshold: float = 0.6,
                           dark_sat: float = 0.8, dark_bright: float = -0.10,
                           chroma_adjust: str = 'none') -> vs.VideoNode:
    """Desaturate and darken dark pixels on all frames (no scene-change gating).

    Wrapper around vs_sc_chroma_bright_tweak with scenechange=False.
    """
    return vs_sc_chroma_bright_tweak(clip, black_threshold, white_threshold, dark_sat, dark_bright,
                                     scenechange=False, chroma_adjust=chroma_adjust)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Direct color mapping using the "chroma adjustment".
"""


def vs_sc_colormap(clip: vs.VideoNode = None, colormap: str = 'none', scenechange: bool = True) -> vs.VideoNode:
    """Apply a direct hue/colour mapping to a clip with optional scene-change gating.

    Delegates to _vs_sc_colormap using image_chroma_tweak.

    :param clip:        RGB24 input clip.
    :param colormap:    Chroma adjustment string (e.g. "300:360|0.8,0.1"). 'none' = bypass.
    :param scenechange: If True, process only scene-change frames. Default True.
    :return:            Colour-mapped RGB24 clip.
    """
    clip_m = _vs_sc_colormap(clip=clip, colormap=colormap, scenechange=scenechange)

    return clip_m


def vs_colormap(clip: vs.VideoNode = None, colormap: str = 'none') -> vs.VideoNode:
    """Apply a direct hue/colour mapping to all frames (no scene-change gating).

    Wrapper around vs_sc_colormap with scenechange=False.

    :param clip:     RGB24 input clip.
    :param colormap: Chroma adjustment string. 'none' = bypass.
    :return:         Colour-mapped RGB24 clip.
    """
    return vs_sc_colormap(clip, colormap, scenechange=False)


def _vs_sc_colormap(clip: vs.VideoNode = None, colormap: str = 'none', scenechange: bool = False) -> vs.VideoNode:
    """Internal implementation of the colormap filter using std.ModifyFrame + image_chroma_tweak.

    :param clip:        RGB24 input clip.
    :param colormap:    Chroma adjustment string. 'none' = bypass.
    :param scenechange: If True, process only scene-change frames.
    :return:            Colour-mapped RGB24 clip.
    """
    def merge_frame(n, f, chroma_adjust: str = 'none', scenechange: bool = True):

        if scenechange:
            is_scenechange = (n == 0) or (f.props.get('_SceneChangePrev', 0) == 1)
            if not is_scenechange:
                return f.copy()

        img = frame_to_image(f)
        img_m = image_chroma_tweak(img, hue_adjust=chroma_adjust)

        return image_to_frame(img_m, f.copy())

    clip_new = clip.std.ModifyFrame(clips=clip,
                                    selector=partial(merge_frame, chroma_adjust=colormap, scenechange=scenechange))
    # clip_new = debug_ModifyFrame(82228, 82232, clip, clips=[clip],
    #                             selector=partial(merge_frame, chroma_adjust=colormap, scenechange=scenechange))
    return clip_new


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Filter used to dark more the dark scenes. The amount of darkness is controlled 
by the parameter dark_amount, while the selected area is controlled by the parameter
dark_threshold.
"""


def vs_sc_dark_tweak(clip: vs.VideoNode = None, dark_threshold: float = 0.3, dark_amount: float = 0.8,
                     scenechange: bool = True,
                     dark_hue_adjust: str = 'none') -> vs.VideoNode:
    """Darken and desaturate dark scene areas with optional scene-change gating.

    Converts dark_threshold and dark_amount to luma/saturation parameters and builds a
    gradient mask from 0.1 to dark_threshold to blend the darkened frame with the original.

    :param clip:            RGB24 input clip.
    :param dark_threshold:  Luma threshold for dark-area selection [0, 1]. Default 0.3.
    :param dark_amount:     Degree of darkening/desaturation [0, 1]. Default 0.8.
    :param scenechange:     If True, process only scene-change frames. Default True.
    :param dark_hue_adjust: Hue range for targeted darkening. 'none' = all hues.
    :return:                Tweaked RGB24 clip.
    """
    d_threshold = 0.1
    d_white_thresh = min(max(dark_threshold, d_threshold), 0.50)
    d_sat = min(max(1.1 - dark_amount, 0.10), 0.80)
    d_bright = -min(max(dark_amount, 0.20), 0.90)

    def merge_frame(n, f, dark_limit: float = 0.3, white_limit: float = 0.6, dark_bright: float = -0.10,
                    dark_sat: float = 0.8, scenechange: bool = True, dark_hue_adjust: str = 'none'):

        if scenechange:
            is_scenechange = (n == 0) or (f.props.get('_SceneChangePrev', 0) == 1)
            if not is_scenechange:
                return f.copy()

        img1 = frame_to_image(f)
        img2 = image_tweak(img1, bright=dark_bright, sat=dark_sat, hue_range=dark_hue_adjust)
        if dark_limit == white_limit:
            img_m = image_luma_merge(img2, img1, dark_limit)
        else:
            img_m = w_image_luma_merge(img2, img1, dark_limit, white_limit)
        return image_to_frame(img_m, f.copy())

    return clip.std.ModifyFrame(clips=clip,
                                selector=partial(merge_frame, dark_limit=d_threshold, white_limit=d_white_thresh,
                                                 dark_bright=d_bright, dark_sat=d_sat, scenechange=scenechange,
                                                 dark_hue_adjust=dark_hue_adjust))


def vs_dark_tweak(clip: vs.VideoNode = None, dark_threshold: float = 0.3, dark_amount: float = 0.8,
                  dark_hue_adjust: str = 'none') -> vs.VideoNode:
    """Darken and desaturate dark scene areas on all frames (no scene-change gating).

    Wrapper around vs_sc_dark_tweak with scenechange=False.
    """
    return vs_sc_dark_tweak(clip, dark_threshold, dark_amount, scenechange=False, dark_hue_adjust=dark_hue_adjust)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
This function force the average luma of a video clip to don't be below the value
defined by the parameter "luma_min". The function allow to modify the gamma
of the clip if the average luma is below the parameter "gamma_luma_min"  
"""


def sc_constrained_tweak(clip: vs.VideoNode = None, luma_min: float = 0.1, gamma: float = 1, gamma_luma_min: float = 0,
                         gamma_alpha: float = 0, gamma_min: float = 0.5, scenechange: bool = True) -> vs.VideoNode:
    """Force minimum luma and apply adaptive gamma correction with optional scene-change gating.

    Delegates to luma_adjusted_levels per frame. When luma is below gamma_luma_min,
    gamma is applied (optionally decaying with luma when gamma_alpha != 0).

    :param clip:           RGB24 input clip.
    :param luma_min:       Minimum average luma target (fraction [0, 1]). Default 0.1.
    :param gamma:          Gamma exponent applied when luma < gamma_luma_min. Default 1 (no-op).
    :param gamma_luma_min: Luma threshold below which gamma activates. Default 0 (disabled).
    :param gamma_alpha:    Power for adaptive gamma decay. Default 0 (constant).
    :param gamma_min:      Minimum gamma value. Default 0.5.
    :param scenechange:    If True, process only scene-change frames. Default True.
    :return:               Luma-adjusted RGB24 clip.
    """
    def change_frame(n, f, luma_min: float = 0.1, gamma: float = 1, gamma_luma_min: float = 0,
                     gamma_alpha: float = 0, gamma_min: float = 0.5, scenechange: bool = True):

        if scenechange:
            is_scenechange = (n == 0) or (f.props.get('_SceneChangePrev', 0) == 1)
            if not is_scenechange:
                return f.copy()

        img = frame_to_image(f)
        img_m = luma_adjusted_levels(img, luma_min, gamma, gamma_luma_min, gamma_alpha, gamma_min)

        return image_to_frame(img_m, f.copy())

    clipm = clip.std.ModifyFrame(clips=clip, selector=partial(change_frame, luma_min=luma_min, gamma=gamma,
                                                              gamma_luma_min=gamma_luma_min, gamma_alpha=gamma_alpha,
                                                              gamma_min=gamma_min, scenechange=scenechange))

    return clipm


def constrained_tweak(clip: vs.VideoNode = None, luma_min: float = 0.1, gamma: float = 1, gamma_luma_min: float = 0,
                      gamma_alpha: float = 0, gamma_min: float = 0.5) -> vs.VideoNode:
    """Force minimum luma and adaptive gamma on all frames (no scene-change gating).

    Wrapper around sc_constrained_tweak with scenechange=False.
    """
    return sc_constrained_tweak(clip, luma_min, gamma, gamma_luma_min, gamma_alpha, gamma_min, scenechange=False)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
video clip tweak function, that allow to change the hue, saturation and brigh, 
with the support of scene change detection.
"""


def vs_sc_tweak(clip: vs.VideoNode = None, hue: float = 0, sat: float = 1, cont: float = 1.0, bright: float = 0,
                gamma: float = 1.0, scenechange: bool = True) -> vs.VideoNode:
    """Adjust hue, saturation, brightness, contrast, and gamma with optional scene-change gating.

    When scenechange=False delegates directly to vs_tweak (YUV-based, faster).
    When scenechange=True processes only scene-change frames via ModifyFrame.

    :param clip:        RGB24 input clip.
    :param hue:         Hue rotation in degrees [-180, +180]. Default 0.
    :param sat:         Saturation multiplier [0, 10]. Default 1.
    :param cont:        Contrast factor [0, 10]. Default 1.0.
    :param bright:      Brightness offset added to luma [-255, +255]. Default 0.
    :param gamma:       Gamma exponent [-10, +10]. Default 1.0.
    :param scenechange: If True, process only scene-change frames. Default True.
    :return:            Adjusted RGB24 clip.
    """
    if hue == 0 and sat == 1 and cont == 1 and bright == 0 and gamma == 1:
        return clip  # non changes

    if not scenechange:
        return vs_tweak(clip, hue, sat, bright, cont, gamma)

    def merge_frame(n, f, hue: float = 0, sat: float = 1, cont: float = 1.0, bright: float = 0, gamma: float = 1.0,
                    scenechange: bool = True):

        if scenechange:
            is_scenechange = (n == 0) or (f.props['_SceneChangePrev'] == 1)
            if not is_scenechange:
                return f.copy()

        img = frame_to_image(f)
        img_m = image_tweak(img, cont=cont, bright=bright, sat=sat, gamma=gamma, hue=hue)

        return image_to_frame(img_m, f.copy())

    return clip.std.ModifyFrame(clips=clip,
                                selector=partial(merge_frame, hue=hue, sat=sat, cont=cont, bright=bright, gamma=gamma,
                                                 scenechange=scenechange))


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Wrapper to vapoursynth function Merge.     
"""


def vs_simple_merge(clipa: vs.VideoNode, clipb: vs.VideoNode, weight: float = 0.5) -> vs.VideoNode:
    """Thin wrapper around std.Merge: result = clipa * (1 - weight) + clipb * weight.

    Returns clipa unchanged when weight==0 and clipb when weight==1.

    :param clipa:  Base clip.
    :param clipb:  Overlay clip.
    :param weight: Weight assigned to clipb [0, 1]. Default 0.5.
    :return:       Merged clip.
    """
    # convert the format for Merge to YUV 8bits

    # A zero weight means that clipa is returned unchanged and 1 means that clipb is returned unchanged
    if weight == 0:
        return clipa
    if weight == 1:
        return clipb

    return vs.core.std.Merge(clipa=clipa, clipb=clipb, weight=weight)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
This function is an extension of the Tweak() function available in Hybrid with
the possibility to change also the gamma of a video clip. It can adjust:
hue, saturation, brightness, contrast and gamma of a video clip.     
"""


def vs_tweak(clip: vs.VideoNode, hue: float = 0, sat: float = 1, bright: float = 0, cont: float = 1, gamma: float = 1,
             coring: bool = False) -> vs.VideoNode:
    """Pre/post - process filter for adjust: hue, saturation, brightness, contrast and gamma of a video clip

    :param clip:      Clip to process. Only RGB24 format is supported.
    :param hue:       Adjust the color hue of the image.
                          hue>0.0 shifts the image towards red.
                          hue<0.0 shifts the image towards green.
                      Range -180.0 to +180.0, default 0.0
    :param sat:       Adjust the color saturation of the image by controlling gain of the color channels.
                          sat>1.0 increases the saturation.
                          sat<1.0 reduces the saturation.
                      Use sat=0 to convert to GreyScale.
                      Range 0.0 to 10.0, default 1.0
    :param bright:    Change the brightness of the image by applying a constant bias to the luma channel.
                            bright>0.0 increases the brightness.
                            bright<0.0 decreases the brightness.
                      Range -255.0 to 255.0, default 0.0
    :param cont:      Change the contrast of the image by multiplying the luma values by a constant.
                            cont>1.0 increase the contrast (the luma range will be stretched).
                            cont<1.0 decrease the contrast (the luma range will be contracted).
                      Range 0.0 to 10.0, default 1.0
    :param gamma:     Change the gamma of image which controls the degree of non-linearity in the luma
                      correction. Higher gamma brightens the output; lower gamma darkens the output.
                      Range -10.0 to 10.0, default 1.0
    :param coring     When set to true, the luma (Y) and chroma are clipped to TV-range;
                      When set to false (the default), the luma and chroma are unconstrained.
    """
    if hue == 0 and sat == 1 and bright == 0 and cont == 1 and gamma == 1:
        return clip  # non changes

    if gamma != 1:
        clip = clip.std.Levels(gamma=gamma)

    c = vs.core

    # convert the format for tweak to YUV 8bits
    clip = clip.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")

    if -1.0 < bright < 1.0:
        bright = bright * 255.0  # normalized to 255 = 2^8-1

    if (hue != 0 or sat != 1) and clip.format.color_family != vs.GRAY:

        hue = hue * math.pi / 180.0
        hue_sin = math.sin(hue)
        hue_cos = math.cos(hue)

        gray = 128 << (clip.format.bits_per_sample - 8)

        chroma_min = 0
        chroma_max = (2 ** clip.format.bits_per_sample) - 1
        if coring:
            chroma_min = 16 << (clip.format.bits_per_sample - 8)
            chroma_max = 240 << (clip.format.bits_per_sample - 8)

        expr_u = "x {} - {} * y {} - {} * + {} + {} max {} min".format(gray, hue_cos * sat, gray, hue_sin * sat, gray,
                                                                       chroma_min, chroma_max)
        expr_v = "y {} - {} * x {} - {} * - {} + {} max {} min".format(gray, hue_cos * sat, gray, hue_sin * sat, gray,
                                                                       chroma_min, chroma_max)

        if clip.format.sample_type == vs.FLOAT:
            expr_u = "x {} * y {} * + -0.5 max 0.5 min".format(hue_cos * sat, hue_sin * sat)
            expr_v = "y {} * x {} * - -0.5 max 0.5 min".format(hue_cos * sat, hue_sin * sat)

        src_u = clip.std.ShufflePlanes(planes=1, colorfamily=vs.GRAY)
        src_v = clip.std.ShufflePlanes(planes=2, colorfamily=vs.GRAY)

        dst_u = c.std.Expr(clips=[src_u, src_v], expr=expr_u)
        dst_v = c.std.Expr(clips=[src_u, src_v], expr=expr_v)

        clip = c.std.ShufflePlanes(clips=[clip, dst_u, dst_v], planes=[0, 0, 0], colorfamily=clip.format.color_family)

    if bright != 0 or cont != 1:

        if clip.format.sample_type == vs.INTEGER:
            luma_lut = []

            luma_min = 0
            luma_max = (2 ** clip.format.bits_per_sample) - 1
            if coring:
                luma_min = 16 << (clip.format.bits_per_sample - 8)
                luma_max = 235 << (clip.format.bits_per_sample - 8)

            for i in range(2 ** clip.format.bits_per_sample):
                val = int((i - luma_min) * cont + bright + luma_min + 0.5)
                luma_lut.append(min(max(val, luma_min), luma_max))

            clip = clip.std.Lut(planes=0, lut=luma_lut)
        else:
            expression = "x {} * {} + 0.0 max 1.0 min".format(cont, bright)

            clip = clip.std.Expr(expr=[expression, "", ""])

    # convert the clip format for HAVC to RGB24
    clip_rgb = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full", dither_type="error_diffusion")

    return clip_rgb


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function to copy the luma of video Clip "orig" in the video "clip" 
"""


def vs_sc_recover_clip_luma(orig: vs.VideoNode = None, clip: vs.VideoNode = None, scenechange: bool = False,
                            sc_framedir: str = None, ref_ext: str = DEF_EXPORT_FORMAT,
                            ref_jpg_quality: int = DEF_JPG_QUALITY) -> vs.VideoNode:
    """Replace the luma (Y) of clip with the luma from orig, preserving clip's chroma.

    Optionally exports the merged frames at scene-change positions to sc_framedir.

    :param orig:           Source of luma (RGB24); determines sharpness/detail.
    :param clip:           Source of chroma (RGB24); coloured output.
    :param scenechange:    If True, export scene-change frames to disk. Default False.
    :param sc_framedir:    Directory for exported reference frames. None = no export.
    :param ref_ext:        File extension for exported frames (e.g. 'png', 'jpg').
    :param ref_jpg_quality: JPEG quality when ref_ext is 'jpg'. Default DEF_JPG_QUALITY.
    :return:               RGB24 clip with luma from orig and chroma from clip.
    """
    def copy_luma_frame(n, f, sc_framedir: str, ref_ext: str, ref_jpg_quality: int):

        img_orig = frame_to_image(f[0])
        img_clip = frame_to_image(f[1])
        img_m = chroma_post_process(img_clip, img_orig)

        if scenechange:
            is_scenechange = (n == 0) or (f.props.get('_SceneChangePrev', 0) == 1)
        else:
            is_scenechange = False

        # orig_prv = f[0].props['_SceneChangePrev']
        # orig_next = f[0].props['_SceneChangeNext']

        # col_prv = f[1].props['_SceneChangePrev']
        # col_next = f[1].props['_SceneChangeNext']

        if not (sc_framedir is None) and is_scenechange:
            img_path = os.path.join(sc_framedir, f"ref_{n:06d}.{ref_ext}")
            if ref_ext == "jpg":
                img_m.save(img_path, subsampling=0, quality=ref_jpg_quality)
            else:
                img_m.save(img_path)

        return image_to_frame(img_m, f[0].copy())

    clip = clip.std.ModifyFrame(clips=[orig, clip], selector=partial(copy_luma_frame, sc_framedir=sc_framedir,
                                                                     ref_ext=ref_ext, ref_jpg_quality=ref_jpg_quality))

    return clip


def vs_recover_clip_luma(orig: vs.VideoNode = None, clip: vs.VideoNode = None) -> vs.VideoNode:
    """Replace the luma of clip with the luma from orig (no export, no scene-change check).

    Wrapper around vs_sc_recover_clip_luma with scenechange=False.

    :param orig: Source of luma (RGB24).
    :param clip: Source of chroma (RGB24).
    :return:     RGB24 clip with Y from orig and UV from clip.
    """
    return vs_sc_recover_clip_luma(orig, clip, scenechange=False)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function to remove noise/grain from clip, strenght control the amount of noise/grain removed, 
if = 0 the filter is not applied. It is based on function KNLMeansCL() with GPU suppot enabled.
"""


def vs_degrain(clip: vs.VideoNode = None, strength: int = 1, device_id: int = 0) -> vs.VideoNode:
    """Remove luma noise/grain using KNLMeansCL with GPU acceleration.

    Converts to YUV420P8, applies KNLMeansCL on the Y plane, converts back to RGB24.

    :param clip:      RGB24 input clip.
    :param strength:  Denoise strength [1, 5]; maps to KNLMeansCL h/d parameters. 0 = bypass.
    :param device_id: GPU device ordinal for KNLMeansCL. Default 0.
    :return:          Denoised RGB24 clip.
    """
    if strength == 0:
        return clip

    match strength:
        case 1:
            dstr = 0.5
            dtmp = 1
        case 2:
            dstr = 1.0
            dtmp = 1
        case 3:
            dstr = 1.5
            dtmp = 1
        case 4:
            dstr = 2.5
            dtmp = 1
        case 5:
            dstr = 3.5
            dtmp = 2
        case _:
            raise vs.Error("HybridAVC: not supported strength value: " + str(strength))

    clip = clip.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")
    clip = vs.core.knlm.KNLMeansCL(clip=clip, d=dtmp, a=2, s=4, h=dstr, channels='Y', device_type="gpu",
                                   device_id=device_id)
    clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_s="full", dither_type="error_diffusion")

    return clip


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: ONLY FOR TESTING
------------------------------------------------------------------------------- 
This filter return a mask calculated on luma range..
"""


def vs_luma_mask(clip: vs.VideoNode = None, luma_mask_limit: float = 0.4,
                 luma_white_limit: float = 0.7) -> vs.VideoNode:
    """Return a clip of luma-based masks (for testing/debugging only).

    :param clip:            RGB24 input clip.
    :param luma_mask_limit: Luma threshold for full selection. Default 0.4.
    :param luma_white_limit: Luma threshold for no selection (gradient in between). Default 0.7.
    :return:                Clip of binary/gradient mask images.
    """
    def mask_frame(n, f, luma_limit: float = 0.4, white_limit: float = 0.7):
        img1 = frame_to_image(f)
        if luma_limit == white_limit:
            # vs.core.log_message(2, "frame[" + str(n) + "]: luma_limit = " + str(luma_limit))
            img_masked = image_luma_merge(img1, img1, luma_limit, True)
        else:
            img_masked = w_image_luma_merge(img1, img1, luma_limit, white_limit, True)
        return image_to_frame(img_masked, f.copy())

    clipm = clip.std.ModifyFrame(clips=clip,
                                 selector=partial(mask_frame, luma_limit=luma_mask_limit, white_limit=luma_white_limit))
    return clipm


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: ONLY FOR TESTING
------------------------------------------------------------------------------- 
Vapoursynth version of AdaptiveLumaMerge (very slow).
"""


def vs_adaptive_Merge(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None,
                      clipb_weight: float = 0.0) -> vs.VideoNode:
    """VapourSynth-native adaptive luma merge (for testing only; very slow).

    Computes per-frame luma via PlaneStats and merges clips with an adaptive weight.
    Use AdaptiveLumaMerge in mcomb.py for production use.

    :param clipa:        Base clip.
    :param clipb:        Overlay clip.
    :param clipb_weight: Maximum weight for clipb. Default 0.0.
    :return:             Merged clip.
    """
    # Vapoursynth version
    def merge_frame(n, f, core, clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, clipb_weight: float = 0.0):
        clip1 = clipa[n]
        clip2 = clipb[n]
        clip2_yuv = clip2.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="full")
        clip2_avg_y = vs.core.std.PlaneStats(clip2_yuv, plane=0)
        luma = clip2_avg_y.get_frame(0).props['PlaneStatsAverage']
        # vs.core.log_message(2, "Luma(" + str(n) + ") = " + str(luma))
        brightness = min(1.5 * luma, 1)
        w = max(clipb_weight * brightness, 0.15)
        clip3 = core.std.Merge(clip1, clip2, weight=w)
        f_out = f.copy()
        f_out = clip3.get_frame(0)
        return f_out

    clipm = clipa.std.ModifyFrame(clips=clipa, selector=partial(merge_frame, core=vs.core, clipa=clipa, clipb=clipb,
                                                                clipb_weight=clipb_weight))
    return clipm


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: the function takes a video clip as an input and calculates the average 
color values for each of the three color planes (red, green, blue).
Then the function is used to adjust the white balance of the input clip based on the 
color balance of the individual frames.
------------------------------------------------------------------------------- 
"""

def vs_rgb_normalize(clip: vs.VideoNode) -> vs.VideoNode:
    """Auto white-balance a clip by normalising each RGB plane to the same average level.

    Per-frame PlaneStats are used to compute per-channel gain factors that equalise the
    red, green, and blue averages to the maximum of the three.

    :param clip: RGB24 input clip.
    :return:     White-balanced RGB24 clip.
    """
    rgb_clip = clip
    r_avg = vs.core.std.PlaneStats(rgb_clip, plane=0)
    g_avg = vs.core.std.PlaneStats(rgb_clip, plane=1)
    b_avg = vs.core.std.PlaneStats(rgb_clip, plane=2)

    def auto_white_adjust(n, f, clip, core):
        small_number = 0.000000001
        red = f[0].props['PlaneStatsAverage']
        green = f[1].props['PlaneStatsAverage']
        blue = f[2].props['PlaneStatsAverage']
        max_rgb = max(red, green, blue)
        red_corr = max_rgb / max(red, small_number)
        green_corr = max_rgb / max(green, small_number)
        blue_corr = max_rgb / max(blue, small_number)
        norm = max(blue,
                   math.sqrt(red_corr * red_corr + green_corr * green_corr + blue_corr * blue_corr) / math.sqrt(3),
                   small_number)
        r_gain = red_corr / norm
        g_gain = green_corr / norm
        b_gain = blue_corr / norm
        return core.std.Expr(clip,
                             expr=['x ' + repr(r_gain) + ' *', 'x ' + repr(g_gain) + ' *', 'x ' + repr(b_gain) + ' *'])

    return vs.core.std.FrameEval(rgb_clip, partial(auto_white_adjust, clip=rgb_clip, core=vs.core),
                                 prop_src=[r_avg, g_avg, b_avg])
