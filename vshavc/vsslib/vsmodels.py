"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2026-05-23
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
module containing the main functions to colorize the frames with deoldify() and ddcolor().
"""
import vapoursynth as vs
import math
import torch
from functools import partial

from vshavc.deoldify.visualize import ModelImageRender
#from vshavc.deoldify.visualize import *
from vshavc.remaster import vs_sc_remaster_colorize
from vshavc.vsslib.imfilters import image_weighted_merge
from vshavc.vsslib.vsfilters import vs_sc_tweak, sc_constrained_tweak, vs_sc_adjust_clip_hue, vs_recover_clip_luma
from vshavc.vsslib.vsutils import frame_to_image, image_to_frame, frame_to_np_array, np_array_to_frame
from vshavc.vsslib.vsutils import debug_ModifyFrame
from vshavc.colormnet import vs_colormnet_remote, vs_colormnet_local
from vshavc.colormnet2 import vs_colormnet2_remote, vs_colormnet2_local
from vshavc.deepex import deepex_colorizer, ModelColorizer
from vshavc.colorization import ModelColorization
from vshavc.havc_utils import rgb_denoise, vs_auto_levels
from vshavc.colormnet2 import vs_colormnet2dit_remote, vs_colormnet2dit_local
from vshavc.vsslib.vsimage_engine import HAVCditEngine

from vshavc.vsslib.constants import *

def vs_colormnet(clip: vs.VideoNode, clip_ref: vs.VideoNode, clip_sc: vs.VideoNode, image_size: int = -1,
                 enable_resize: bool = False, frame_propagate: bool = True, render_vivid: bool = True,
                 max_memory_frames: int = 0, encode_mode: int = 0, ref_weight: float = 1.0) -> vs.VideoNode:
    """Colorize a clip using ColorMNet v1 (exemplar-based, flat memory model).

    Routes to the remote (XML-RPC subprocess) or local (in-process) implementation
    based on encode_mode. For local mode (1) the max_memory_frames limit is inferred
    automatically from available GPU RAM when set to 0.

    :param clip:               B&W source clip (RGB24).
    :param clip_ref:           Coloured reference clip carrying scene-change frame props.
    :param clip_sc:            Auxiliary clip used for ref-merge scene detection (may be None).
    :param image_size:         Inference resolution override. -1 = use clip size.
    :param enable_resize:      Enable internal resolution upscaling for higher colour accuracy.
    :param frame_propagate:    If True, non-reference frames are propagated (coloured from neighbours).
    :param render_vivid:       If True, reset frame memory at each reference frame update.
    :param max_memory_frames:  Maximum number of reference frames kept in memory. 0 = auto.
    :param encode_mode:        0 = remote (no GPU limit), 1 = local (GPU-limited),
                               2 = remote all-ref, 3 = local all-ref (testing only).
    :param ref_weight:         Weight used to merge reference frames into the output [0, 1].
    :return:                   Colourised clip (RGB24).
    """
    if encode_mode == 1:
        if max_memory_frames is None or max_memory_frames == 0:
            gpu_mem_free, gpu_mem_total = torch.cuda.mem_get_info()
            mem_tot_k = round(gpu_mem_total / 1024 / 1024 / 1024, 0)
            if mem_tot_k < 8.5:
                max_memory_frames = 4
            elif mem_tot_k < 12.5:
                max_memory_frames = 8
            elif mem_tot_k < 16.5:
                max_memory_frames = 18
            else:
                max_memory_frames = 25

    match encode_mode:
        case 0 | 2:
            return vs_colormnet_remote(clip, clip_ref, clip_sc, image_size, enable_resize, frame_propagate,
                                       render_vivid, max_memory_frames, ref_weight, use_all_refs=(encode_mode == 2))
        case 1 | 3:  # encode_mode = 3 is supported only for testing, given the memory limitation of this method
            return vs_colormnet_local(clip, clip_ref, clip_sc, image_size, enable_resize, frame_propagate,
                                      render_vivid, max_memory_frames, ref_weight, use_all_refs=(encode_mode == 3))
        case _:
            raise vs.Error("HAVC_deepex: unknown encode mode: " + str(encode_mode))


def vs_colormnet2(clip: vs.VideoNode, clip_ref: vs.VideoNode, clip_sc: vs.VideoNode, image_size: int = -1,
                  enable_resize: bool = False, frame_propagate: bool = True, render_vivid: bool = True,
                  max_memory_frames: int = 0, encode_mode: int = 0, ref_weight: float = 1.0,
                  sc_framedir: str = None, retry_perm_share_threshold: float = 0.25,
                  retry_model: int = 0, backbone: str = "dinov3") -> vs.VideoNode:
    """Colorize a clip using ColorMNet v2 (CMNET2) with a sliding permanent-memory window.

    max_memory_frames is interpreted as the sliding window size (DEF_XRF_WINDOW_SIZE by
    default) rather than a flat memory limit. Routes to remote (0) or local (1) backend.

    :param clip:               B&W source clip (RGB24).
    :param clip_ref:           Coloured reference clip carrying scene-change frame props.
    :param clip_sc:            Auxiliary clip for ref-merge scene detection (may be None).
    :param image_size:         Inference resolution override. -1 = use clip size.
    :param enable_resize:      Enable internal upscaling for higher colour accuracy.
    :param frame_propagate:    If True, propagate colours to non-reference frames.
    :param render_vivid:       If True, increase output saturation by ~15%.
    :param max_memory_frames:  Sliding window size (number of reference frames). 0 = DEF_XRF_WINDOW_SIZE.
    :param encode_mode:        0 = remote (XML-RPC subprocess), 1 = local (in-process).
    :param ref_weight:         Blend weight for reference frames [0, 1].
    :param sc_framedir:        Path to a directory of pre-saved reference images for direct access (ref_mode=0).
    :param retry_perm_share_threshold:  Threshold on perm_share below which reference_frame_missing() returns True.
                                        Default 0.25.
    :param retry_model:         Model used by the retry path to colorize missing reference frames.
                                     0 = HAVC (DeOldify + DDColor), 1 = DiT fp4, 2 = DiT int4. Default 0.
    :param backbone:            Key-encoder backbone: 'dinov3' (default, native PyTorch implementation,
                                requires the dinov3 weights in the local weights dir) or 'dinov2'.
    :return:                   Colourised clip (RGB24).
    """
    # max_memory_frames acts as window_size for the sliding perm_mem; default to DEF_XRF_WINDOW_SIZE
    if max_memory_frames is None or max_memory_frames == 0:
        max_memory_frames = DEF_XRF_WINDOW_SIZE

    match encode_mode:
        case 0:
            return vs_colormnet2_remote(clip, clip_ref, clip_sc, image_size, enable_resize, frame_propagate,
                                        render_vivid, max_memory_frames, ref_weight, sc_framedir,
                                        retry_perm_share_threshold=retry_perm_share_threshold,
                                        retry_model=retry_model, backbone=backbone)
        case 1:
            return vs_colormnet2_local(clip, clip_ref, clip_sc, image_size, enable_resize, frame_propagate,
                                       render_vivid, max_memory_frames, ref_weight, sc_framedir,
                                       retry_perm_share_threshold=retry_perm_share_threshold,
                                       retry_model=retry_model, backbone=backbone)
        case _:
            raise vs.Error(f"HAVC_cmnet2: encode_mode must be 0 or 1, got {encode_mode}")

# Default parameters for HAVCditEngine when the caller does not provide them.
# Keys mirror HAVCditEngine.__init__() arguments exactly.
_DEF_DIT_ENGINE_PARAMS: dict = {
    "host"                 : "127.0.0.1",
    "port"                 : 8765,
    "model_name"           : "nunchaku-qwen",
    "model_precision"      : "fp4",
    "model_rank"           : "32",
    "model_inference_steps": "4",
    "cache_dir"            : "",
    "full_model_path"      : "",
    "prompt"               : (
        "Colorize this image, natural colors. "
        "Strictly preserve all shapes, edges and background details."
    ),
    "steps"   : 2,
    "img_size": 0,
}

def vs_colormnet2dit(clip: vs.VideoNode, clip_ref: vs.VideoNode,
                     dit_engine_params: dict = None,  image_size: int = -1,
                     enable_resize: bool = False, frame_propagate: bool = False,
                     render_vivid: bool = False, max_memory_frames: int = 0,
                     encode_mode: int = 0, retry_perm_share_threshold: float = 0.0,
                     retry_model: int = 0, backbone: str = "dinov3") -> vs.VideoNode:
    """Colorize a clip using CMNET2-DIT with a sliding permanent-memory window.

    DIT variant of vs_colormnet2(): reference frames are treated as B&W and
    colorized by HAVCditEngine (a DiT-based model via RPC) before being loaded
    into CMNET2 permanent memory.  Colorization always runs in pairs to exploit
    HAVCditEngine.colorize_image_pair(); a single colorize_image() call handles
    any odd leftover at the end of the reference list.

    max_memory_frames is automatically rounded down to the nearest even number
    and used as the sliding window size (default: DEF_XRF_WINDOW_SIZE).
    Routes to remote (encode_mode=0) or local (encode_mode=1) CMNET2 backend.

    :param clip:                        B&W source clip (RGB24).
    :param clip_ref:                    B&W clip carrying scene-change props and
                                        reference frames.  In HAVC_cmnet2dit this
                                        is derived from the input clip itself.
    :param dit_engine_params:           Dict of keyword arguments forwarded to
                                        HAVCditEngine().  Missing keys fall back to
                                        _DEF_DIT_ENGINE_PARAMS defaults.
                                        Recognised keys (all optional):
                                            host, port, model_name, model_precision,
                                            model_rank, model_inference_steps,
                                            cache_dir, full_model_path,
                                            prompt, steps, img_size.
    :param image_size:                  CMNET2 inference resolution override (-1 = clip size).
    :param enable_resize:               Enable internal CMNET2 upscaling.
    :param frame_propagate:             If True, propagate colours from reference frames.
    :param render_vivid:                If True, apply a gentle hue/saturation boost (~15%).
    :param max_memory_frames:           Sliding window size.  0 → DEF_XRF_WINDOW_SIZE.
                                        Rounded down to nearest even number internally.
    :param encode_mode:                 0 = remote CMNET2 backend (XML-RPC subprocess).
                                        1 = local CMNET2 backend (in-process).
    :param retry_perm_share_threshold:  CMNET2 retry threshold (0.0 = disabled, default).
    :param retry_model:                 If retry_perm_share_threshold > 0, model used to colorize missing (default: 1)
                                        reference frames. Allowed values are:
                                             0 = HAVC (DeOldify + DDColor),
                                             1 = DiT fp4,
                                             2 = DiT int4.
    :param backbone:                    Key-encoder backbone: 'dinov3' (default, native PyTorch
                                        implementation, requires the dinov3 weights in the local
                                        weights dir) or 'dinov2'.
    :return:                            Colourised clip (RGB24).
    """
    # Resolve window size: default → DEF_XRF_WINDOW_SIZE, then force even.
    if max_memory_frames is None or max_memory_frames == 0:
        max_memory_frames = DEF_XRF_WINDOW_SIZE
    max_memory_frames = max(2, math.trunc(max_memory_frames / 2) * 2)

    # Build HAVCditEngine parameters: start from defaults, overlay caller's dict.
    engine_params = dict(_DEF_DIT_ENGINE_PARAMS)
    if dit_engine_params:
        engine_params.update(dit_engine_params)

    dit_engine = HAVCditEngine(**engine_params)

    match encode_mode:
        case 0:
            return vs_colormnet2dit_remote(
                clip, clip_ref, dit_engine,
                image_size, enable_resize, frame_propagate,
                render_vivid, max_memory_frames,
                retry_perm_share_threshold=retry_perm_share_threshold,
                retry_model=retry_model, backbone=backbone)
        case 1:
            return vs_colormnet2dit_local(
                clip, clip_ref, dit_engine,
                image_size, enable_resize, frame_propagate,
                render_vivid, max_memory_frames,
                retry_perm_share_threshold=retry_perm_share_threshold,
                retry_model=retry_model, backbone=backbone)
        case _:
            raise vs.Error(f"HAVC_cmnet2dit: encode_mode must be 0 or 1, got {encode_mode}")


def vs_deepex(clip: vs.VideoNode, clip_ref: vs.VideoNode, clip_sc: vs.VideoNode, image_size: list = [432, 768],
              enable_resize: bool = False, wls_filter_on: bool = True, render_vivid: bool = True,
              propagate: bool = True,
              ref_weight: float = 1.0) -> vs.VideoNode:
    """Colorize a clip using Deep-Exemplar-based Video Colorization.

    At each scene-change frame the reference image is updated; all other frames are
    colorized by propagation. When ref_weight < 1 and clip_sc is provided, non-scene-
    change frames are blended with the reference frame at the given weight.

    :param clip:           B&W source clip (RGB24).
    :param clip_ref:       Coloured reference clip with scene-change frame props.
    :param clip_sc:        Auxiliary clip for ref-merge scene detection (may be None).
    :param image_size:     [H, W] inference resolution. Default [432, 768].
    :param enable_resize:  Enable internal upscaling for higher colour accuracy.
    :param wls_filter_on:  Apply WLS (Weighted Least Squares) post-filter. Default True.
    :param render_vivid:   Increase output saturation by ~25%. Default True.
    :param propagate:      Propagate colours from reference to non-reference frames. Default True.
    :param ref_weight:     Blend weight between colorized output and reference frame [0, 1].
    :return:               Colourised clip (RGB24).
    """
    colorizer = deepex_colorizer(image_size=image_size, enable_resize=enable_resize)

    def deepex_clip_color_merge(n, f, colorizer: ModelColorizer = None, wls_on: bool = True,
                                propagate: bool = True, render_vivid: bool = True,
                                weight: float = 1.0) -> vs.VideoFrame:

        is_scenechange = f[2].props['_SceneChangePrev'] == 1
        is_scenechange_ext = is_scenechange and f[2].props['_SceneChangeNext'] == 1
        img_orig = frame_to_image(f[0])
        img_ref = frame_to_image(f[1])

        if n == 0:
            colorizer.set_ref_frame(img_ref)
        elif is_scenechange:
            # vs.core.log_message(2, "Reference Frame: " + str(n))
            frame_as_video = not is_scenechange_ext and propagate
            colorizer.set_ref_frame(img_ref, frame_propagate=frame_as_video)

        img_color = colorizer.colorize_frame(img_orig, wls_filter_on=wls_on, render_vivid=render_vivid)

        # the frames that are not scenechange are merged with the ref frames generated by HAVC
        # this should stabilize further the colors generated with HAVC.
        if not is_scenechange:
            img_color_m = image_weighted_merge(img_color, img_ref, weight)
        else:  # the frame obtained from a reference should be already good is merged with low weight
            img_color_m = img_color   # image_weighted_merge(img_color, img_ref, 0.20)

        return image_to_frame(img_color_m, f[0].copy())

    def deepex_clip_color(n, f, colorizer: ModelColorizer = None, wls_on: bool = True,
                          propagate: bool = True, render_vivid: bool = True) -> vs.VideoFrame:

        is_scenechange = f[1].props['_SceneChangePrev'] == 1
        is_scenechange_ext = f[1].props['_SceneChangeNext'] == 1
        img_orig = frame_to_image(f[0])

        if n == 0:
            img_ref = frame_to_image(f[1])
            # vs.core.log_message(2, "Reference Frame: " + str(n))
            colorizer.set_ref_frame(img_ref)
        elif is_scenechange:
            img_ref = frame_to_image(f[1])
            # vs.core.log_message(2, "Reference Frame: " + str(n))
            frame_as_video = not is_scenechange_ext and propagate
            colorizer.set_ref_frame(img_ref, frame_as_video)

        img_color = colorizer.colorize_frame(img_orig, wls_filter_on=wls_on, render_vivid=render_vivid)

        return image_to_frame(img_color, f[0].copy())

    if 0 < ref_weight < 1 and not (clip_sc is None):
        clip_colored = clip.std.ModifyFrame(clips=[clip, clip_ref, clip_sc],
                                            selector=partial(deepex_clip_color_merge, colorizer=colorizer,
                                                             wls_on=wls_filter_on, render_vivid=render_vivid,
                                                             propagate=propagate, weight=ref_weight))
    else:
        clip_colored = clip.std.ModifyFrame(clips=[clip, clip_ref],
                                            selector=partial(deepex_clip_color, colorizer=colorizer,
                                                             propagate=propagate,
                                                             wls_on=wls_filter_on, render_vivid=render_vivid))
    return clip_colored


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to DeepRemaster. 
"""


def vs_deepremaster(clip: vs.VideoNode, clip_ref: vs.VideoNode, clip_sc: vs.VideoNode, render_vivid: bool = True,
                    ref_weight: float = 1.0, ref_size: int = 256, frame_size: int = 320, memory_size: int = None,
                    ref_frequency: int = 0, device_index: int = 0) -> vs.VideoNode:
    """Colorize a clip using DeepRemaster (temporal source-reference attention network).

    Delegates to vs_sc_remaster_colorize with the provided parameters. Unlike ColorMNet,
    DeepRemaster stores full reference images in a buffer, enabling accurate long-term
    colour consistency at the cost of higher GPU memory usage.

    :param clip:          B&W source clip (RGB24).
    :param clip_ref:      Coloured reference clip with scene-change frame props.
    :param clip_sc:       Auxiliary clip for ref-merge scene detection (may be None).
    :param render_vivid:  Increase output saturation by ~20%. Default True.
    :param ref_weight:    Blend weight for reference frames [0, 1].
    :param ref_size:      Minimum edge size of reference frames used for inference. Default 256.
    :param frame_size:    Minimum edge size of input frames used for inference. Default 320.
    :param memory_size:   Reference frame buffer size. None/0 = DEF_NUM_RF_FRAMES.
    :param ref_frequency: Minimum reference frame insertion frequency (frames). Default 0.
    :param device_index:  GPU device ordinal. -1 = CPU mode.
    :return:              Colourised clip (RGB24).
    """
    if memory_size is None or memory_size == 0:
        memory_size = DEF_NUM_RF_FRAMES
    if memory_size < DEF_MIN_RF_FRAMES:
        memory_size = DEF_MIN_RF_FRAMES

    clip_colored = vs_sc_remaster_colorize(clip, clip_ref, clip_sc=clip_sc, length=DEF_BATCH_SIZE,
                                           render_vivid=render_vivid, ref_minedge=ref_size,
                                           frame_mindim=frame_size, merge_weight=ref_weight,
                                           ref_buffer_size=memory_size, ref_frequency=ref_frequency,
                                           device_index=device_index)

    return clip_colored


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to deoldify. 
"""


def vs_deoldify(clip: vs.VideoNode, method: int = 2, model: int = 0, render_factor: int = 24, scenechange: bool = True,
                package_dir: str = "") -> vs.VideoNode:
    """Colorize every frame of a clip using DeOldify (scene-change detection disabled).

    Thin wrapper around vs_sc_deoldify that forces scenechange=False so all frames
    are colourised unconditionally.

    :param clip:          B&W RGB24 input clip.
    :param method:        Combination mode (0 = DeOldify only, 1 = skip, 2 = merge).
    :param model:         DeOldify model (0 = Video, 1 = Stable, 2 = Artistic).
    :param render_factor: Render factor controlling inference resolution [10, 44].
    :param scenechange:   Ignored; always False for this function.
    :param package_dir:   Path to the vshavc package directory (for model weights).
    :return:              Colourised RGB24 clip.
    """
    return vs_sc_deoldify(clip, method, model, render_factor, scenechange=False, package_dir=package_dir)


def vs_sc_deoldify(clip: vs.VideoNode, method: int = 2, model: int = 0, render_factor: int = 24,
                   scenechange: bool = True, package_dir: str = "") -> vs.VideoNode | None:
    """Colorize a clip using DeOldify with optional scene-change-only processing.

    When scenechange=True only frames flagged as scene changes (via _SceneChangePrev)
    are processed; other frames are passed through unchanged. Returns None if method==1
    (DDColor-only mode).

    :param clip:          B&W RGB24 input clip.
    :param method:        0 = DeOldify only, 1 = return None (DDColor only), 2+ = merge mode.
    :param model:         0 = ColorizeVideo_gen, 1 = ColorizeStable_gen, 2 = ColorizeArtistic_gen.
    :param render_factor: Render factor [10, 44]. Default 24.
    :param scenechange:   If True, process only scene-change frames.
    :param package_dir:   Path to the vshavc package directory (for model weights).
    :return:              Colourised RGB24 clip, or None when method==1.
    """
    if method == 1:
        return None

    match model:
        case 0:  # model: Video
            colorizer = ModelImageRender(package_dir=package_dir, modelname='video', render_factor=render_factor)
        case 1:  # model: Stable
            colorizer = ModelImageRender(package_dir=package_dir, modelname='stable', render_factor=render_factor,
                                         video_weight=DEF_STABLE_WEIGHT)
        case 2:  # model: Artistic
            colorizer = ModelImageRender(package_dir=package_dir, modelname='artistic', render_factor=render_factor,
                                         video_weight=DEF_ARTISTIC_WEIGHT)
        case _:
            colorizer = ModelImageRender(package_dir=package_dir, modelname='video', render_factor=render_factor)

    clipa_rgb = _deoldify(clip, colorizer, scenechange=scenechange)

    return clipa_rgb


def _deoldify(clip: vs.VideoNode, colorizer: ModelImageRender = None, scenechange: bool = True) -> vs.VideoNode:
    """Internal per-frame DeOldify colorization via std.ModifyFrame.

    :param clip:        B&W RGB24 input clip.
    :param colorizer:   Initialised ModelImageRender instance.
    :param scenechange: If True, process only scene-change frames; else process all frames.
    :return:            Colourised RGB24 clip.
    """
    def deoldify_colorize(n: int, f: vs.VideoFrame, colorizer: ModelImageRender, scenechange: bool) -> vs.VideoFrame:

        if scenechange:
            is_scenechange = (n == 0) or (f.props['_SceneChangePrev'] == 1)
            if not is_scenechange:
                return f.copy()

        img_orig = frame_to_image(f)

        img_color = colorizer.get_transformed_image(img_orig, post_process=True)

        return image_to_frame(img_color, f.copy())

    return clip.std.ModifyFrame(clips=[clip],
                                selector=partial(deoldify_colorize, colorizer=colorizer, scenechange=scenechange))


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to Colorization. 
"""


def vs_sc_colorization(clip: vs.VideoNode, colorizer_model: str = 'siggraph17',
                       scenechange: bool = True, frame_size:int = 256) -> vs.VideoNode:
    """Colorize a clip using Zhang et al.'s SIGGRAPH17 or ECCV16 model.

    :param clip:             B&W RGB24 input clip.
    :param colorizer_model:  'siggraph17' or 'eccv16'.
    :param scenechange:      If True, process only scene-change frames.
    :param frame_size:       Square inference resolution in pixels. Default 256.
    :return:                 Colourised RGB24 clip.
    """
    m_colorizer = ModelColorization(model=colorizer_model, use_gpu=True)
    f_size = frame_size # min(frame_size, 512)

    def colorization(n: int, f: vs.VideoFrame, colorizer: ModelColorization = None,
                     scflag: bool = True, f_size: int = 256) -> vs.VideoFrame:

        if scflag:
            is_scenechange = (n == 0) or (f.props['_SceneChangePrev'] == 1)
            if not is_scenechange:
                return f.copy()

        np_frame_orig = frame_to_np_array(f)

        np_frame_colored = colorizer.colorize_frame(np_frame_orig)

        return np_array_to_frame(np_frame_colored, f.copy())

    """ 
    clip_new = debug_ModifyFrame(f_start=0, f_end=500, clip=clip, clips=[clip],
                                 selector=partial(colorization, colorizer=m_colorizer, scflag=scenechange))
    """
    clip_new = clip.std.ModifyFrame(clips=[clip], selector=partial(colorization, colorizer=m_colorizer,
                                    scflag=scenechange, f_size=f_size))
    #"""

    return clip_new


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to function ddcolor() with tweak pre-process.
"""

def vs_ddcolor(clip: vs.VideoNode, method: int = 2, model: int = 1, render_factor: int = 24,
               tweaks_flags: list[bool] = (False, False, False),
               tweaks: list = (DEF_TWEAK_p, "none"),
               enable_fp16: bool = True, device_index: int = 0, num_streams: int = 1) -> vs.VideoNode:
    """Colorize every frame using DDColor (scene-change detection disabled).

    Thin wrapper around vs_sc_ddcolor with scenechange=False.

    :param clip:           B&W RGB24 input clip.
    :param method:         0 = DeOldify only (return None), 1 = DDColor only, 2+ = merge mode.
    :param model:          DDColor model: 0 = ModelScope, 1 = Artistic, 2 = Siggraph17, 3 = ECCV16.
    :param render_factor:  Render factor controlling inference resolution [10, 64]. Default 24.
    :param tweaks_flags:   (tweaks_enabled, denoise_enabled, retinex_enabled).
    :param tweaks:         Tweak parameters list; see vs_sc_ddcolor for details.
    :param enable_fp16:    Use FP16 (RGBH) for DDColor inference. Default True.
    :param device_index:   GPU device ordinal. Default 0.
    :param num_streams:    Number of CUDA streams. Default 1.
    :return:               Colourised RGB24 clip.
    """
    return vs_sc_ddcolor(clip, method, model, render_factor, tweaks_flags, tweaks, enable_fp16, scenechange=False,
                         device_index=device_index, num_streams=num_streams)


def vs_sc_ddcolor(clip: vs.VideoNode, method: int = 2, model: int = 1, render_factor: int = 24,
                  tweaks_flags: list[bool] = (False, False, False),
                  tweaks: list = (DEF_TWEAK_p, "none"),
                  enable_fp16: bool = True, scenechange: bool = True, device_index: int = 0,
                  num_streams: int = 1) -> vs.VideoNode | None:
    """Colorize a clip using DDColor with optional pre-processing tweaks and scene-change support.

    Applies brightness/contrast/gamma tweaks or Retinex pre-processing before DDColor
    inference. Supports models 0/1 (DDColor) and 2/3 (Zhang Siggraph17/ECCV16). Optionally
    applies RGB denoise and hue adjustment post-processing. Returns None when method==0.

    :param clip:           B&W RGB24 input clip.
    :param method:         0 = return None (DeOldify only), 1+ = use DDColor.
    :param model:          0 = ModelScope, 1 = Artistic, 2 = Siggraph17, 3 = ECCV16.
    :param render_factor:  Render factor [10, 64]. Default 24.
    :param tweaks_flags:   (tweaks_enabled, denoise_enabled, retinex_enabled).
    :param tweaks:         Tweak parameter list or pair (DEF_TWEAK_p, hue_adjust).
    :param enable_fp16:    Use FP16 (RGBH) for inference. Default True.
    :param scenechange:    If True, process only scene-change frames.
    :param device_index:   GPU device ordinal. Default 0.
    :param num_streams:    Number of CUDA streams. Default 1.
    :return:               Colourised RGB24 clip, or None when method==0.
    """
    if method == 0:
        return None

    if model in (0, 1):
        import vsddcolor

    # input size must a multiple of 32
    input_size = math.trunc(render_factor / 2) * 32

    # unpack tweaks
    tweaks_enabled = tweaks_flags[0]
    denoise_enabled = tweaks_flags[1]
    retinex_enabled = tweaks_flags[2]

    if len(tweaks) == 2:
        bright = tweaks[0][0]
        cont = tweaks[0][1]
        gamma = tweaks[0][2]
        luma_constrained_tweak = tweaks[0][3]
        luma_min = tweaks[0][4]
        gamma_luma_min = tweaks[0][5]
        gamma_alpha = tweaks[0][6]
        gamma_min = tweaks[0][7]
        hue_adjust = tweaks[1].lower()
    else:
        bright = tweaks[0]
        cont = tweaks[1]
        gamma = tweaks[2]
        luma_constrained_tweak = tweaks[3]
        luma_min = tweaks[4]
        gamma_luma_min = tweaks[5]
        gamma_alpha = tweaks[6]
        gamma_min = tweaks[7]
        if len(tweaks) > 8:
            hue_adjust = tweaks[8]
        else:
            hue_adjust = 'none'

    if tweaks_enabled:
        if retinex_enabled:
            clipb = vs_auto_levels(clip, mode='strong', method=5, luma_blend=True, range_tv=True)
        elif luma_constrained_tweak:
            clipb = vs_sc_tweak(clip, bright=bright, cont=cont,
                                scenechange=scenechange)  # contrast and bright are adjusted before the constrainded luma and gamma
            clipb = sc_constrained_tweak(clipb, luma_min=luma_min, gamma=gamma, gamma_luma_min=gamma_luma_min,
                                         gamma_alpha=gamma_alpha, gamma_min=gamma_min, scenechange=scenechange)
        else:
            clipb = vs_sc_tweak(clip, bright=bright, cont=cont, gamma=gamma, scenechange=scenechange)
    else:
        clipb = clip

    if model > 1:
        if model == 2:
            clipb_rgb = vs_sc_colorization(clipb, colorizer_model='siggraph17', scenechange=scenechange, frame_size=input_size)
        else:
            clipb_rgb = vs_sc_colorization(clipb, colorizer_model='eccv16', scenechange=scenechange, frame_size=input_size)
    else:
        # adjusting clip's color space to RGBH for vsDDColor
        if enable_fp16:
            clipb = vsddcolor.ddcolor(clipb.resize.Bicubic(format=vs.RGBH, range_s="full"), model=model,
                                      input_size=input_size, scenechange=scenechange, device_index=device_index,
                                      num_streams=num_streams)
        else:
            clipb = vsddcolor.ddcolor(clipb.resize.Bicubic(format=vs.RGBS, range_s="full"), model=model,
                                      input_size=input_size, scenechange=scenechange, device_index=device_index,
                                      num_streams=num_streams)

        # adjusting color space to RGB24 for deoldify
        clipb_rgb = clipb.resize.Bicubic(format=vs.RGB24, range_s="full")

    if hue_adjust != 'none':
        clipb_rgb = vs_sc_adjust_clip_hue(clipb_rgb, hue_adjust, scenechange=scenechange)

    if denoise_enabled:  # remove rgb noise
        clipb_rgb = rgb_denoise(clipb_rgb, denoise_levels=[0.3, 0.2], rgb_factors=[0.98, 1.02, 1.0])

    if tweaks_enabled:
        return vs_recover_clip_luma(clip, clipb_rgb)
    else:
        return clipb_rgb
