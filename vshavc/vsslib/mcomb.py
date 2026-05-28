"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2026-05-02
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
module containing the functions used to combine deoldify() and ddcolor().
"""
import vapoursynth as vs
import math
import os
from PIL import Image
from functools import partial
from typing import Tuple

from vshavc.vsslib.imfilters import image_luma_merge, w_image_luma_merge, image_weighted_merge, get_image_luma
from vshavc.vsslib.imfilters import chroma_stabilizer, image_tweak, chroma_stabilizer_adaptive
from vshavc.vsslib.vsfilters import vs_tweak, vs_sc_recover_clip_color, vs_sc_recover_gradient_color
from vshavc.vsslib.vsfilters import vs_sc_recover_clip_luma, vs_simple_merge
from vshavc.vsslib.vsutils import HAVC_LogMessage, MessageType, frame_to_image, image_to_frame, get_ref_images

from vshavc.vsslib.constants import *

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function to build the refrence image used for the inference by DeepEx
"""


def vs_ext_reference_clip(clip: vs.VideoNode, sc_framedir: str = None, clip_resize: bool = False) -> vs.VideoNode:
    """Build a reference clip from images stored in sc_framedir for exemplar-based models.

    At scene-change frames the corresponding image (named ref_NNNNNN.ext) is read
    from sc_framedir and injected into the clip; other frames pass through unchanged.
    Optionally resizes the input clip to match the reference image dimensions.

    :param clip:        B&W or coloured RGB24 input clip carrying scene-change frame props.
    :param sc_framedir: Path to the directory containing reference image files.
    :param clip_resize: If True, resize the clip to match the first reference image size.
    :return:            Clip with reference images injected at scene-change positions.
    """
    if not os.path.exists(sc_framedir):
        HAVC_LogMessage(MessageType.EXCEPTION, "vs_ext_reference_clip(): frames path '", sc_framedir, "' is invalid")

    ref_images = get_ref_images(sc_framedir)

    if not ref_images:
        HAVC_LogMessage(MessageType.EXCEPTION, "vs_ext_reference_clip(): no reference images found in '",
                        sc_framedir, "'")
    ref_images.sort()

    f_size = (clip.width, clip.height)
    if clip_resize:
        # resize the clip to the same size of first reference image
        img_path = ref_images[0]
        if os.path.isfile(img_path):
            try:
                ref_img = Image.open(img_path).convert('RGB')
                if ref_img.size != f_size:
                    clip = clip.resize.Spline64(width=ref_img.size[0], height=ref_img.size[1])
                    f_size = ref_img.size
            except Exception as error:
                HAVC_LogMessage(MessageType.WARNING, "Error reading reference frame: ", img_path, " -> ", error)

    def set_clip_frame(n, f, img_list: list = None, f_size: Tuple[int, int] = None):

        f_out = f.copy()
        is_scenechange = (n == 0) or (f.props['_SceneChangePrev'] == 1)

        f_out.props['_SceneChangePrev'] = 0   # set scene-change detection OFF by default

        if not is_scenechange:
            return f_out

        img_name = f"ref_{n:06d}"
        ext_ref_img = [f for f in img_list if img_name in f]
        if len(ext_ref_img) > 0:
            img_path = ext_ref_img[0]
            if os.path.isfile(img_path):
                try:
                    ref_img = Image.open(img_path).convert('RGB')
                    if ref_img.size != f_size:
                        ref_img = ref_img.resize(f_size, Image.Resampling.LANCZOS)
                        # vs.core.log_message(2, "Resized reference frame: " + img_path + " size= " + str(f_size))
                        f_out.props['_SceneChangePrev'] = 1  # set scene-change detection ON
                except Exception as error:
                    HAVC_LogMessage(MessageType.WARNING, "Error reading reference frame: ", img_path, " -> ", error)
                    return f_out
            else:
                HAVC_LogMessage(MessageType.WARNING, "vs_ext_reference_clip(): path '", img_path, "' is invalid")
                return f_out
        else:
            HAVC_LogMessage(MessageType.WARNING, "vs_ext_reference_clip(): not found file: '", img_name, ".*' ")
            return f_out

        if ref_img is None:
            return f_out

        return image_to_frame(ref_img, f.copy())

    clip_ref = clip.std.ModifyFrame(clips=[clip], selector=partial(set_clip_frame, img_list=ref_images, f_size=f_size))

    #clip_ref = debug_ModifyFrame(50, 80, clip, clips=[clip],
    #                             selector=partial(set_clip_frame, img_list=ref_images, f_size=f_size))

    return clip_ref


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
main function used to combine the colored images with deoldify() and ddcolor()
"""


def vs_combine_models(clip_a: vs.VideoNode = None, clip_b: vs.VideoNode = None, method: int = 0, sat: list = (1, 1),
                      hue: list = (0, 0), clipb_weight: float = 0.5, CMC_p: list = DEF_CMC_p, LMM_p: list = DEF_LMM_p,
                      ALM_p: list = DEF_ALM_p, CRT_p: list = DEF_CRT_p,
                      invert_clips: bool = False) -> vs.VideoNode:
    """Combine DeOldify (clip_a) and DDColor (clip_b) clips using the selected merge method.

    Convenience wrapper around vs_sc_combine_models with scenechange=False (all frames
    are processed). See vs_sc_combine_models for full parameter documentation.
    """
    return vs_sc_combine_models(clip_a, clip_b, method, sat, hue, clipb_weight, CMC_p, LMM_p,
                                ALM_p, CRT_p, invert_clips, scenechange=False)


def vs_sc_combine_models(clip_a: vs.VideoNode = None, clip_b: vs.VideoNode = None, method: int = 0, sat: list = (1, 1),
                      hue: list = (0, 0), clipb_weight: float = 0.5, CMC_p: list = DEF_CMC_p, LMM_p: list = DEF_LMM_p,
                      ALM_p: list = DEF_ALM_p, CRT_p: list = DEF_CRT_p, invert_clips: bool = False,
                      scenechange: bool = True) -> vs.VideoNode:
    """Combine DeOldify (clip_a) and DDColor (clip_b) using one of 6 merge strategies.

    Applies per-clip sat/hue tweaks before merging. Dispatches to the appropriate
    merge function based on method (2–7). With scenechange=True only scene-change
    frames are processed.

    :param clip_a:       Stable reference clip (e.g. DeOldify). RGB24.
    :param clip_b:       Colourful clip (e.g. DDColor). RGB24.
    :param method:       Merge method: 2=Simple, 3=ConstrainedChroma, 4=LumaMasked,
                         5=AdaptiveLuma, 6=ChromaRetention, 7=ChromaBoundAdaptive.
    :param sat:          (sat_a, sat_b) saturation multipliers applied before merging.
    :param hue:          (hue_a, hue_b) hue rotations applied before merging.
    :param clipb_weight: Blend weight assigned to clip_b [0, 1].
    :param CMC_p:        Parameters for ConstrainedChroma / ChromaBoundAdaptive merges.
    :param LMM_p:        Parameters for LumaMasked merge.
    :param ALM_p:        Parameters for AdaptiveLuma merge.
    :param CRT_p:        Parameters for ChromaRetention merge.
    :param invert_clips: If True, swap clip_a and clip_b roles.
    :param scenechange:  If True, process only scene-change frames.
    :return:             Merged RGB24 clip.
    """
    # vs.core.log_message(2, "combine_models: method=" + str(method) + ", clipa = " + str(clipa) + ", clipb = " + str(clipb))

    # unpack combine_params
    chroma_threshold = CMC_p[0]
    if len(CMC_p) > 1:
        red_fix: bool = CMC_p[1]
        base_tol: int = CMC_p[2]
        max_extra: int = CMC_p[3]
    else:
        red_fix: bool = True
        base_tol: int = 20
        max_extra: int = 24
    luma_mask_limit = LMM_p[0]
    luma_white_limit = LMM_p[1]
    luma_mask_sat = LMM_p[2]
    luma_threshold = ALM_p[0]
    alpha = ALM_p[1]
    min_weight = ALM_p[2]
    crt_sat = CRT_p[0]
    crt_tht = CRT_p[1]
    crt_alpha = CRT_p[2]
    crt_resize = CRT_p[3]
    crt_mask_weight = CRT_p[4]
    crt_algo = CRT_p[5]

    if invert_clips:
        clipa = clip_b
        clipb = clip_a
    else:
        clipa = clip_a
        clipb = clip_b

    if clipa is not None:
        clipa = vs_tweak(clipa, hue=hue[0], sat=sat[0])
        if clipb is None:
            return clipa

    if clipb is not None:
        clipb = vs_tweak(clipb, hue=hue[1], sat=sat[1])
        if clipa is None:
            return clipb

    if method == 2:
        return SimpleMerge(clipa, clipb, clipb_weight, scenechange=scenechange)
    if method == 3:
        clip_ccm = ConstrainedChromaMerge(clipa, clipb, clipb_weight, chroma_threshold, red_fix, scenechange=scenechange)
        clip_m = SimpleMerge(clipa, clipb, min(clipb_weight, 0.6), scenechange=scenechange)
        clip_ccmm = SimpleMerge(clip_ccm, clip_m, clipb_weight=0.3, scenechange=scenechange)
        return clip_ccmm
    if method == 4:
        return LumaMaskedMerge(clipa, clipb, luma_mask_limit, luma_white_limit, luma_mask_sat, clipb_weight,
                               scenechange=scenechange)
    if method == 5:
        return AdaptiveLumaMerge(clipa, clipb, luma_threshold, alpha, clipb_weight, min_weight, scenechange=scenechange)

    if method == 6:
        return ChromaRetentionMerge(clipa, clipb, sat=crt_sat, tht=crt_tht, clipb_weight=clipb_weight, alpha=crt_alpha,
                                    mask_weight=crt_mask_weight, scenechange=scenechange, chroma_resize=crt_resize,
                                    algo=crt_algo)
    if method == 7:
        return ChromaBoundAdaptiveMerge(clipa, clipb, red_fix=red_fix, base_tol = base_tol, max_extra = max_extra,
                                              clipb_weight = clipb_weight, scenechange = scenechange)
    else:
        raise vs.Error("HAVC: only dd_method in (0,6) is supported")


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
the images are combined using a weighted merge, where the parameter clipb_weight
represent the weight assigned to the colors provided by ddcolor() 
"""


def SimpleMerge(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, clipb_weight: float = 0.5,
                  scenechange: bool = False) -> vs.VideoNode:
    """Merge two clips with a fixed per-frame weighted blend.

    :param clipa:        Base clip (e.g. DeOldify). RGB24.
    :param clipb:        Overlay clip (e.g. DDColor). RGB24.
    :param clipb_weight: Weight assigned to clipb [0, 1]. Default 0.5.
    :param scenechange:  If True, process only scene-change frames. Default False.
    :return:             Merged RGB24 clip.
    """
    def merge_frame(n, f, weight: float = 0.5, scenechange: bool = True):

        if scenechange:
            is_scenechange = (n == 0) or (f[0].props['_SceneChangePrev'] == 1)
            if not is_scenechange:
                return f[0].copy()

        img1 = frame_to_image(f[0])
        img2 = frame_to_image(f[1])
        img_m = image_weighted_merge(img1, img2, weight)
        return image_to_frame(img_m, f[0].copy())

    clipm = clipa.std.ModifyFrame(clips=[clipa, clipb],
                                  selector=partial(merge_frame, weight=clipb_weight, scenechange=scenechange))

    return clipm


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
the clips are combined using a mask merge, the pixels of clipb with luma < luma_mask_limit
will be filled with the pixels of clipa, if the parameter clipm_weight > 0
the masked image will be merged with clipa 
"""


def LumaMaskedMerge(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, luma_mask_limit: float = 0.4,
                    luma_white_limit: float = 0.7, luma_mask_sat=1.0, clipm_weight: float = 0.5,
                    scenechange: bool = False) -> vs.VideoNode:
    """Merge two clips using a luma-based mask: dark pixels use clipa, bright pixels use clipb.

    A gradient mask is built from luma_mask_limit to luma_white_limit. Optionally the
    masked result is blended back with clipa at clipm_weight.

    :param clipa:           Base clip (dark-pixel source). RGB24.
    :param clipb:           Overlay clip (bright-pixel source). RGB24.
    :param luma_mask_limit: Luma below which clipa is used (fraction [0, 1]). Default 0.4.
    :param luma_white_limit: Luma above which clipb is used; gradient between limits. Default 0.7.
    :param luma_mask_sat:   Saturation applied to clipa in the mask region. Default 1.0.
    :param clipm_weight:    Weight of the masked frame in the final blend with clipa. Default 0.5.
    :param scenechange:     If True, process only scene-change frames. Default False.
    :return:                Merged RGB24 clip.
    """
    if luma_mask_sat < 1:
        # vs.core.log_message(2, "LumaMaskedMerge: mask_sat = " + str(luma_mask_sat))
        clipc = vs_tweak(clipa, sat=luma_mask_sat)
    else:
        clipc = clipa

    def merge_frame(n, f, weight: float, luma_limit: float, white_limit: float, scenechange: bool):

        if scenechange:
            is_scenechange = (n == 0) or (f[0].props['_SceneChangePrev'] == 1)
            if not is_scenechange:
                return f[0].copy()

        img1 = frame_to_image(f[0])
        img2 = frame_to_image(f[1])
        img3 = frame_to_image(f[2])
        if luma_limit == white_limit:
            # vs.core.log_message(2, "frame[" + str(n) + "]: luma_limit = " + str(luma_limit))
            img_masked = image_luma_merge(img3, img2, luma_limit)
        else:
            img_masked = w_image_luma_merge(img3, img2, luma_limit, white_limit)
        if clipm_weight < 1.0:
            img_m = image_weighted_merge(img1, img_masked, weight)
        else:
            img_m = img_masked
        return image_to_frame(img_m, f[0].copy())

    clipm = clipa.std.ModifyFrame(clips=[clipa, clipb, clipc],
                                  selector=partial(merge_frame, weight=clipm_weight, luma_limit=luma_mask_limit,
                                                   white_limit=luma_white_limit, scenechange=scenechange))
    return clipm


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
given the ddcolor() perfomance is quite bad on dark scenes, the images are 
combinaed by decreasing the weight assigned to ddcolor() when the luma is 
below a given threshold given by: luma_threshold. 
For example with: luma_threshold = 0.6 and alpha = 1, the weight assigned to 
ddcolor() will start to decrease linearly when the luma < 60% till "min_weight".
For alpha=2, begins to decrease quadratically.      
"""


def AdaptiveLumaMerge(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, luma_threshold: float = 0.6,
                      alpha: float = 1.0, clipb_weight: float = 0.5, min_weight: float = 0.15,
                      scenechange: bool = False) -> vs.VideoNode:
    """Merge two clips with a clipb weight that decreases for dark frames.

    ``w = max(clipb_weight * (luma / luma_threshold)^alpha, min_weight)``
    when frame luma < luma_threshold; otherwise w = clipb_weight.

    :param clipa:           Base clip (e.g. DeOldify). RGB24.
    :param clipb:           Overlay clip (e.g. DDColor). RGB24.
    :param luma_threshold:  Luma level below which weight starts decreasing [0, 1]. Default 0.6.
    :param alpha:           Power-law exponent. 1 = linear, 2 = quadratic. Default 1.0.
    :param clipb_weight:    Maximum weight for clipb [0, 1]. Default 0.5.
    :param min_weight:      Minimum weight for clipb on very dark frames. Default 0.15.
    :param scenechange:     If True, process only scene-change frames. Default False.
    :return:                Merged RGB24 clip.
    """
    def merge_frame(n, f, luma_limit: float, min_w: float, alpha: float, weight: float, scenechange: bool):

        if scenechange:
            is_scenechange = (n == 0) or (f[0].props['_SceneChangePrev'] == 1)
            if not is_scenechange:
                return f[0].copy()

        img1 = frame_to_image(f[0])
        img2 = frame_to_image(f[1])
        luma = get_image_luma(img2)
        if luma < luma_limit:
            bright_scale = pow(luma / luma_limit, alpha)
            w = max(weight * bright_scale, min_w)
        else:
            w = weight
        # vs.core.log_message(2, "Luma(" + str(n) + ") = " + str(luma) + ", weight = " + str(w))
        img_m = Image.blend(img1, img2, w)
        return image_to_frame(img_m, f[0].copy())

    clipm = clipa.std.ModifyFrame(clips=[clipa, clipb],
                                  selector=partial(merge_frame, luma_limit=luma_threshold, min_w=min_weight,
                                                   alpha=alpha, weight=clipb_weight, scenechange=scenechange))
    return clipm


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
given that the colors provided by deoldify() are more conservative and stable 
than the colors obtained with ddcolor() images are combined by assigning
a limit to the amount of difference in chroma values between deoldify() and
ddcolor() this limit is defined by the parameter threshold. The limit is applied
to the image converted to "YUV". For example when threshold=0.1, the chroma
values "U","V" of ddcolor() image will be constrained to have an absolute
percentage difference respect to "U","V" provided by deoldify() not higher than 10%    
"""


def ConstrainedChromaMerge(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, clipb_weight: float = 0.5,
                           chroma_threshold: float = 0.2, red_fix: bool = True,
                           scenechange: bool = False) -> vs.VideoNode:
    """Merge two clips by limiting clipb's chroma to ±chroma_threshold of clipa's chroma.

    Uses chroma_stabilizer to constrain U/V channel differences. Optionally applies a
    red-shift correction on dark frames (luma < 0.3) to avoid reddish artefacts.

    :param clipa:             Stable reference clip (e.g. DeOldify). RGB24.
    :param clipb:             Colourful clip (e.g. DDColor). RGB24.
    :param clipb_weight:      Final blend weight of constrained clipb [0, 1]. Default 0.5.
    :param chroma_threshold:  Maximum allowed fractional chroma deviation. Default 0.2 (20%).
    :param red_fix:           Apply graduated desaturation on dark frames. Default True.
    :param scenechange:       If True, process only scene-change frames. Default False.
    :return:                  Merged RGB24 clip.
    """
    def merge_frame(n, f, level: float, redfix: bool, weight: float, scenechange: bool):

        if scenechange:
            is_scenechange = (n == 0) or (f[0].props['_SceneChangePrev'] == 1)
            if not is_scenechange:
                return f[0].copy()

        img1 = frame_to_image(f[0])
        img2 = frame_to_image(f[1])
        img_stab = chroma_stabilizer(img1, img2, level, weight)

        if not redfix:
            return image_to_frame(img_stab, f[0].copy())

        luma = get_image_luma(img_stab, 255)
        # Dark frames red-shift adjustment
        if luma > 0.3:
            img_m = img_stab
        elif luma > 0.2:
            img_dark = image_tweak(img_stab, sat=0.9, hue_range="280:360,0:30")
            img_m = w_image_luma_merge(img_dark, img_stab, 0.2, 0.3)
        elif luma > 0.1:
            img_dark = image_tweak(img_stab, sat=0.8, hue_range="280:360,0:30")
            img_m = w_image_luma_merge(img_dark, img_stab, 0.1, 0.2)
        else:
            img_m = image_tweak(img_stab, sat=0.7)
        return image_to_frame(img_m, f[0].copy())

    clipm = clipa.std.ModifyFrame(clips=[clipa, clipb],
                                  selector=partial(merge_frame, level=chroma_threshold, redfix=red_fix,
                                                   weight=clipb_weight, scenechange=scenechange))
    return clipm


def ChromaBoundAdaptiveMerge(
        clipa: vs.VideoNode,
        clipb: vs.VideoNode,
        red_fix: bool = True,
        base_tol: int = 14,
        max_extra: int = 18,
        clipb_weight: float = 0.5,
        scenechange: bool = False
) -> vs.VideoNode:
    """
    Adaptive version of Constrained-Chroma. In this version the chroma tolerance is adaptive, i.e., it is applied an
    approach that will allow more color variation in textured/complex regions and less in smooth areas.
    The texture strength is computed via Laplacian.

    Args:
        clipa: vs.VideoNode,  # Stable reference (e.g., DeOldify as RGB)
        clipb: vs.VideoNode,  # New colorized clip
        red_fix: bool,  # if True will be applied a correction on the red regions.
        base_tol: int = 14,  # Base chroma tolerance (smooth areas)
        max_extra: int = 18,  # Extra tolerance for textured areas
        clipb_weight: float = 1.0,  # Blending weight (1.0 = full constrained clipb)
        scenechange: bool = False  # Only process on scene changes
    """
    def merge_frame(n: int, f: list[vs.VideoFrame],
                    redfix: bool, base_tol: int, max_extra: int,
                    weight: float, scenechange: bool) -> vs.VideoFrame:

        if scenechange:
            is_scenechange = (n == 0) or (f[0].props.get('_SceneChangePrev', 0) == 1)
            if not is_scenechange:
                return f[0].copy()

        img1 = frame_to_image(f[0])  # clipa
        img2 = frame_to_image(f[1])  # clipb
        img_stab = chroma_stabilizer_adaptive(img1, img2, base_tol=base_tol, max_extra=max_extra, weight=weight)

        if not redfix:
            return image_to_frame(img_stab, f[0].copy())

        luma = get_image_luma(img_stab, 255)
        # Dark frames red-shift adjustment
        if luma > 0.3:
            img_m = img_stab
        elif luma > 0.2:
            img_dark = image_tweak(img_stab, sat=0.9, hue_range="280:360,0:30")
            img_m = w_image_luma_merge(img_dark, img_stab, 0.2, 0.3)
        elif luma > 0.1:
            img_dark = image_tweak(img_stab, sat=0.8, hue_range="280:360,0:30")
            img_m = w_image_luma_merge(img_dark, img_stab, 0.1, 0.2)
        else:
            img_m = image_tweak(img_stab, sat=0.7)

        return image_to_frame(img_m, f[0].copy())

    # Ensure both clips are RGB24
    if clipa.format.id != vs.RGB24:
        clipa = clipa.resize.Bicubic(format=vs.RGB24)
    if clipb.format.id != vs.RGB24:
        clipb = clipb.resize.Bicubic(format=vs.RGB24)

    clipm = clipa.std.ModifyFrame(clips=[clipa, clipb], selector=partial(merge_frame, redfix=red_fix, base_tol=base_tol,
                                                        max_extra=max_extra, weight=clipb_weight,
                                                        scenechange=scenechange))

    #clipm = debug_ModifyFrame(f_start = 6060, f_end = 90300, clip = clipa, clips = [clipa, clipb],
    #                          selector = partial(merge_frame, base_tol=base_tol, max_extra=max_extra, weight=clipb_weight, scenechange=scenechange))

    return clipm

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Given that the colors provided by deoldify() are more conservative and stable 
than the colors obtained with ddcolor(). This function try to restore the 
colors of gray pixels provide by deoldify() by using the colors provided by ddcolor(). 
"""

def ChromaRetentionMerge(clip_a: vs.VideoNode = None, clip_b: vs.VideoNode = None, sat: float = 0.8, tht: int = 30,
                     clipb_weight: float = 0.9, alpha: float = 2.0, mask_weight: float = 0, scenechange: bool = False,
                     chroma_resize: bool = True, return_mask: bool = False, binary_mask: bool = False,
                     algo: int = 0) -> vs.VideoNode:
    """Utility function to restore the colors of gray pixels in clip_a by using the colors provided in clip_b.

        :param clip_a:        clip to repair the colors, only RGB24 format is supported
        :param clip_b:        clip with the colors to restore, only RGB24 format is supported
        :param sat:           this parameter allows to change the saturation of colored clip (default = 0.8)
        :param tht:           threshold to identify gray pixels, range[0, 255] (default = 30)
        :param clipb_weight:  represent the weight of the filtered clip. Range[0,1] (default = 0.9)
        :param alpha:         parameter used to control the steepness of gradient curve, values above the default value
                              will preserve more pixels, but could introduce some artifacts, range[1, 10] (default = 2)
        :param mask_weight:   represent the weight for merging the masked colored clip with clip_a/clip_b:
                                  if weight > 0: merge(clip_restored, clip_b, weight)
                                  if weight < 0: merge(clip_restored, clip_a, weight)
                              Range[0,1], default = 0 (is returned clip_restored without additional merge)
        :param chroma_resize: if True, the frames will be resized to improve the filter speed (default = True)
        :param return_mask:   if True, will be returned the mask used to identify the gray pixels (white region), could
                              be useful to visualize the gradient mask for debugging, (default = false).
        :param binary_mask:   if True, will be used a binary mask instead of a gradient mask, could be useful to get a
                              clear view on the selected desaturated regions for debugging, (default = false)
        :param scenechange:   if True, the filter will be applied only on the scene-change frames, (default = false)
        :param algo:          algorithm to build the mask, allowed values are:
                                  [0] = Linear decay with steep gradient, (default)
                                  [1] = Linear decay
                                  [2] = Exponential decay
    """

    alpha = max(min(alpha, DEF_MAX_COLOR_ALPHA), DEF_MIN_COLOR_ALPHA)

    clip_luma = clip_a
    if chroma_resize and not return_mask:
        rf = min(max(math.trunc(0.4 * clip_luma.width / 16), 16), 48)
        frame_size = min(rf * 16, clip_luma.width)
        if frame_size < clip_luma.width:  # sanity check, avoid upscale
            clip = clip_a.resize.Spline64(width=frame_size, height=frame_size)
            clip_color = clip_b.resize.Spline64(width=frame_size, height=frame_size)
        else:
            clip = clip_a
            clip_color = clip_b
            chroma_resize = False
    else:
        clip = clip_a
        clip_color = clip_b

    clipa_w = mask_weight
    if binary_mask:
        clip_restored = vs_sc_recover_clip_color(clip=clip, clip_color=clip_color, sat=sat, tht=tht, weight=clipa_w,
                                                 tht_scen=1.0, hue_adjust='none', return_mask=return_mask,
                                                 scenechange=scenechange)
    else:
        clip_restored = vs_sc_recover_gradient_color(clip=clip, clip_color=clip_color, sat=sat, tht=tht, weight=clipa_w,
                                                     alpha=alpha, return_mask=return_mask, scenechange=scenechange)

    if return_mask:
        return clip_restored

    # Restore the original size, necessary for merge and chroma_resize
    if chroma_resize:
        clip_restored = clip_restored.resize.Spline64(width=clip_luma.width, height=clip_luma.height)
        clip_restored = vs_sc_recover_clip_luma(clip_luma, clip_restored, scenechange=scenechange)

    clip_restored = vs_simple_merge(clip_luma, clip_restored, weight=clipb_weight)


    return clip_restored

class HAVCimageEngine:
    """
    Singleton orchestrator for single-frame colorization combining DeOldify and
    DDColor with a weighted merge.

    Designed for use cases where a SINGLE B&W frame must be colorized on demand
    inside another VapourSynth filter callback (e.g. CMNET2 retry path with a
    fresh reference). For colorizing entire clips, use vs_sc_deoldify /
    vs_sc_ddcolor / vs_combine_models — they have their own model lifecycle
    scoped to the filter.

    Singleton scope: one engine per process. The DeOldify and DDColor models
    are loaded into GPU memory ONCE, lazily, on first use of colorize_merged().
    Calling HAVCimageEngine() with different parameters after the first call
    has NO EFFECT — the existing instance is returned with its original
    configuration. To switch parameters mid-process, call reset() first
    (not recommended in steady-state).

    Lazy loading: instantiating HAVCimageEngine() does NOT load any model. The
    DeOldify model is loaded on the first colorize_merged() call; the DDColor
    model is loaded on the first DDColorEngine() call inside it. If the engine
    is never used (e.g. retry never triggers in the clip), no VRAM is consumed.

    Typical usage:
        engine = HAVCimageEngine(modelname='video', render_factor=32)
        merged_img = engine.colorize_merged(bw_pil_image)

    Notes:
        - Input PIL image must be RGB ('RGB' mode); B&W input must have the L
          channel replicated into the 3 RGB channels by the caller.
        - Output is PIL RGB at the same resolution as input.
        - DDColor model is hardcoded to 'artistic' (model=1) since it tends to
          give richer colors that better complement DeOldify on hard scenes.
        - Merge weight 0.5 follows HAVC's default for DeOldify+DDColor blending.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, modelname: str = 'video', render_factor: int = 32,
                 ddcolor_model: int = 1, merge_weight: float = 0.5):
        if self._initialized:
            return

        # Validate parameters early — model loading is deferred to first use.
        if modelname not in ('video', 'stable', 'artistic'):
            raise ValueError(
                f"HAVCimageEngine: modelname must be 'video', 'stable' or 'artistic', "
                f"got '{modelname}'")
        if not (10 <= render_factor <= 44):
            raise ValueError(
                f"HAVCimageEngine: render_factor must be in [10, 44], got {render_factor}")
        if ddcolor_model not in (0, 1):
            raise ValueError(
                f"HAVCimageEngine: ddcolor_model must be 0 or 1, got {ddcolor_model}")
        if not (0.0 <= merge_weight <= 1.0):
            raise ValueError(
                f"HAVCimageEngine: merge_weight must be in [0.0, 1.0], got {merge_weight}")

        self._modelname = modelname
        self._render_factor = render_factor
        self._ddcolor_model = ddcolor_model
        self._ddcolor_input_size = render_factor * 16
        self._merge_weight = merge_weight

        # Resolve package_dir from this file's location: mcomb.py is in vsslib/,
        # so the parent directory is the vshavc package root, which is what
        # ModelImageRender expects.
        self._package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        # Lazily-initialised engines.
        self._deoldify_engine = None  # ModelImageRender
        self._ddcolor_engine = None   # vsddcolor.DDColorEngine

        self._initialized = True

    def _ensure_engines(self):
        """Load DeOldify and DDColor models on first call. No-op afterwards."""
        if self._deoldify_engine is None and self._merge_weight != 1:
            # Ensure GPU device is properly configured before loading models.
            # Without this, the model may silently load on CPU (~10x slower).
            from vshavc.deoldify import device
            from vshavc.deoldify.device_id import DeviceId
            if not device.is_gpu():
                device.set(DeviceId.GPU0)

            # Imported lazily to keep the import surface minimal at module load.
            from vshavc.deoldify.visualize import ModelImageRender
            self._deoldify_engine = ModelImageRender(
                package_dir=self._package_dir,
                modelname=self._modelname,
                render_factor=self._render_factor,
            )
        if self._ddcolor_engine is None and self._merge_weight != 0:
            import vsddcolor
            self._ddcolor_engine = vsddcolor.DDColorEngine(
                model=self._ddcolor_model,
                input_size=self._ddcolor_input_size,
            )

    def colorize_merged(self, pil_img: Image.Image) -> Image.Image:
        """
        Colorize a single PIL B&W image as a weighted merge of DeOldify and
        DDColor outputs.

        :param pil_img: B&W input as PIL Image, mode 'RGB' (L replicated 3 times).
        :return:        Merged colorized PIL Image, mode 'RGB', same resolution
                        as input.
        """
        if not isinstance(pil_img, Image.Image):
            raise ValueError("HAVCimageEngine.colorize_merged: input must be a PIL Image")
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        self._ensure_engines()

        if self._merge_weight == 0:
            img_m = self._deoldify_engine.get_transformed_image(pil_img, post_process=True)
        elif self._merge_weight == 1:
            img_m = self._ddcolor_engine.colorize_image(pil_img)
        else:
            img_deoldify = self._deoldify_engine.get_transformed_image(pil_img, post_process=True)
            img_ddcolor = self._ddcolor_engine.colorize_image(pil_img)
            # image_weighted_merge: out = a * (1 - w) + b * w
            # With w=0.5, equally weighted blend.
            img_m = image_weighted_merge(img_deoldify, img_ddcolor, self._merge_weight)

        return img_m

    def reset(self):
        """
        Drop the loaded models and reset the singleton. After reset() the next
        HAVCimageEngine(...) call will reload models fresh, possibly with
        different parameters. Use sparingly.
        """
        # Drop DeOldify reference; relies on Python GC + torch.cuda.empty_cache.
        self._deoldify_engine = None
        # DDColorEngine has its own reset() that handles its singleton.
        if self._ddcolor_engine is not None:
            try:
                self._ddcolor_engine.reset()
            except Exception:
                pass
            self._ddcolor_engine = None

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        HAVCimageEngine._instance = None
        HAVCimageEngine._initialized = False