"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2025-09-28
version:
LastEditors: Dan64
LastEditTime: 2026-04-21
-------------------------------------------------------------------------------
Description: CMNET2 - ColorMNet Extended
             (implemented as Singleton)
-------------------------------------------------------------------------------
main Vapoursynth wrapper for model: CMNET2
URL: https://github.com/dan64/cmnet2
"""
from __future__ import annotations, print_function

from vsdeoldify.colormnet2.colormnet2_render import ColorMNetRender2
from vsdeoldify.colormnet2.colormnet2_utils import *
from vsdeoldify.colormnet2.colormnet2_server import ColorMNetServer2
from vsdeoldify.colormnet2.colormnet2_client import ColorMNetClient2
from vsdeoldify.vsslib.imfilters import image_weighted_merge
from vsdeoldify.vsslib.constants import *
from vsdeoldify.vsslib.vsfilters import vs_tweak
from vsdeoldify.vsslib.vsutils import MessageType, HAVC_LogMessage, debug_ModifyFrame

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# weights are not duplicated
package_dir = path.dirname(os.path.realpath(__file__)).replace("colormnet2", "colormnet")


def vs_colormnet2_local(clip: vs.VideoNode, clip_ref: vs.VideoNode, clip_sc: vs.VideoNode, image_size: int = -1,
                        enable_resize: bool = False, frame_propagate: bool = False, render_vivid: bool = True,
                        max_memory_frames: int = 0, ref_weight: float = 1.0, sc_framedir: str = None,
                        retry_perm_share_threshold: float = 0.30) -> vs.VideoNode:
    vid_length = clip.num_frames

    # max_memory_frames here is the user's window_size; the colorizer's long-term memory stays large
    colorizer = ColorMNetRender2(image_size=image_size, vid_length=vid_length, enable_resize=enable_resize,
                                 encode_mode=1, max_memory_frames=DEF_MAX_MEMORY_FRAMES,
                                 reset_on_ref_update=False, retry_perm_share_threshold=retry_perm_share_threshold,
                                 project_dir=package_dir)

    clip_colored = _colormnet2_async(colorizer, clip, clip_ref, clip_sc, frame_propagate, ref_weight,
                                     max_memory_frames, sc_framedir,
                                     enable_retry=retry_perm_share_threshold > 0)

    if render_vivid:
        clip_colored = vs_tweak(clip_colored, hue=DEF_VIVID_HUE_LOW, sat=DEF_VIVID_SAT_LOW)

    return clip_colored


def _colormnet2_async(colorizer: ColorMNetRender2, clip: vs.VideoNode, clip_ref: vs.VideoNode,
                      clip_sc: vs.VideoNode, frame_propagate: bool = False, ref_weight: float = 1.0,
                      max_memory_frames: int = 0, sc_framedir: str = None, enable_retry: bool = False) -> vs.VideoNode:
    reader: RefImageReader2 = RefImageReader2()
    if sc_framedir is not None:
        reader.load_from_dir(sc_framedir, target_size=(clip.width, clip.height))
    else:
        reader.load_clip_ref(clip_ref, clip_sc, window_size=max_memory_frames)
    perm_mem_win = PermMemWindow(colorizer, reader, window_size=max_memory_frames)
    perm_mem_win.preload_initial()

    def colormnet_clip_color_merge(n, f, perm_mem_win: PermMemWindow, reader: RefImageReader2,
                                   colorizer: ColorMNetRender2 = None,
                                   propagate: bool = False, weight: float = 1.0) -> vs.VideoFrame:

        img_orig = frm_to_img(f[0])
        img_ref = frm_to_img(f[1])  # always same as video to merge, even if frame_as_video = False

        if n == 0:
            colorizer.set_ref_frame(reader.get_ref_image(0), propagate)
        else:
            colorizer.set_ref_frame(None)

        perm_mem_win.adjust(n)
        img_color = colorizer.colorize_frame(ti=n, frame_i=img_orig)

        is_scenechange = f[2].props['_SceneChangePrev'] == 1
        if not is_scenechange:
            img_color_m = image_weighted_merge(img_color, img_ref, weight)
        else:
            img_color_m = img_color

        return img_to_frm(img_color_m, f[0].copy())

    def colormnet_clip_color(n, f, perm_mem_win: PermMemWindow, reader: RefImageReader2,
                             colorizer: ColorMNetRender2 = None,
                             propagate: bool = False, enable_retry: bool = False) -> vs.VideoFrame:

        img_orig = frm_to_img(f[0])

        if n == 0:
            colorizer.set_ref_frame(reader.get_ref_image(0), propagate)
        else:
            colorizer.set_ref_frame(None)

        perm_mem_win.adjust(n)

        if enable_retry:
            # In-process call: retry logic lives in ColorMNetRender2 itself.
            # No RPC overhead because we are running locally (encode_mode=1).
            img_color = colorizer.colorize_frame_with_retry(ti=n, frame_i=img_orig)
        else:
            img_color = colorizer.colorize_frame(ti=n, frame_i=img_orig)

        return img_to_frm(img_color, f[0].copy())

    if 0 < ref_weight < 1 and not (clip_sc is None):
        clip_colored = clip.std.ModifyFrame(clips=[clip, clip_ref, clip_sc],
                                            selector=partial(colormnet_clip_color_merge, perm_mem_win=perm_mem_win,
                                                             reader=reader, colorizer=colorizer,
                                                             propagate=frame_propagate, weight=ref_weight))
    else:
        #"""
        clip_colored = clip.std.ModifyFrame(clips=[clip, clip_ref],
                                            selector=partial(colormnet_clip_color, perm_mem_win=perm_mem_win,
                                                             reader=reader, colorizer=colorizer,
                                                             propagate=frame_propagate, enable_retry=enable_retry))
        """
        clip_colored = debug_ModifyFrame(f_start=0, f_end=500, clip=clip, clips=[clip, clip_ref],
                                         selector=partial(colormnet_clip_color, perm_mem_win=perm_mem_win,
                                                          reader=reader, colorizer=colorizer,
                                                          propagate=frame_propagate))
        """
    return clip_colored


def vs_colormnet2_remote(clip: vs.VideoNode, clip_ref: vs.VideoNode, clip_sc: vs.VideoNode, image_size: int = -1,
                         enable_resize: bool = False, frame_propagate: bool = False, render_vivid: bool = True,
                         max_memory_frames: int = 0, ref_weight: float = 1.0, sc_framedir: str = None,
                         retry_perm_share_threshold: float = 0.25, server_port: int = 0) -> vs.VideoNode:
    vid_length = clip.num_frames

    server = ColorMNetServer2(server_port=server_port).run_server()
    # max_memory_frames here is the user's window_size; the colorizer's long-term memory stays large
    colorizer = ColorMNetClient2(image_size=image_size, vid_length=vid_length, enable_resize=enable_resize,
                                 encode_mode=0, max_memory_frames=DEF_MAX_MEMORY_FRAMES,
                                 reset_on_ref_update=False, retry_perm_share_threshold=retry_perm_share_threshold,
                                 server_port=server.get_port())

    if not colorizer.is_initialized():
        HAVC_LogMessage(MessageType.EXCEPTION, "Failed to initialize ColorMNet[remote] try ColorMNet[local]")

    clip_colored = _colormnet2_client(colorizer, clip, clip_ref, clip_sc, frame_propagate, ref_weight,
                                      max_memory_frames, sc_framedir, enable_retry=retry_perm_share_threshold > 0)

    if render_vivid:
        clip_colored = vs_tweak(clip_colored, hue=DEF_VIVID_HUE_LOW, sat=DEF_VIVID_SAT_LOW)

    return clip_colored


def _colormnet2_client(colorizer: ColorMNetClient2, clip: vs.VideoNode, clip_ref: vs.VideoNode,
                       clip_sc: vs.VideoNode, frame_propagate: bool = False, ref_weight: float = 1.0,
                       max_memory_frames: int = 0, sc_framedir: str = None, enable_retry:bool=False) -> vs.VideoNode:

    reader: RefImageReader2 = RefImageReader2()
    if sc_framedir is not None:
        reader.load_from_dir(sc_framedir, target_size=(clip.width, clip.height))
    else:
        reader.load_clip_ref(clip_ref, clip_sc, window_size=max_memory_frames)
    perm_mem_win = PermMemWindow(colorizer, reader, window_size=max_memory_frames)
    perm_mem_win.preload_initial()

    def colormnet_client_color_merge(n, f, perm_mem_win: PermMemWindow, reader: RefImageReader2,
                                     colorizer: ColorMNetClient2 = None,
                                     propagate: bool = False, weight: float = 1.0) -> vs.VideoFrame:

        img_orig = frm_to_img(f[0])
        img_ref = frm_to_img(f[1])  # always same as video to merge, even if frame_as_video = False

        if n == 0:
            colorizer.set_ref_frame(reader.get_ref_image(0), propagate)
        else:
            colorizer.set_ref_frame(None)

        perm_mem_win.adjust(n)
        img_color = colorizer.colorize_frame(ti=n, frame_i=img_orig)

        is_scenechange = f[2].props['_SceneChangePrev'] == 1
        if not is_scenechange:
            img_color_m = image_weighted_merge(img_color, img_ref, weight)
        else:
            img_color_m = img_color

        return img_to_frm(img_color_m, f[0].copy())


    def colormnet_client_color(n, f, perm_mem_win: PermMemWindow, reader: RefImageReader2,
                               colorizer: ColorMNetClient2 = None, enable_retry: bool = False,
                               propagate: bool = False) -> vs.VideoFrame:

        img_orig = frm_to_img(f[0])

        if n == 0:
            colorizer.set_ref_frame(reader.get_ref_image(0), propagate)
        else:
            colorizer.set_ref_frame(None)

        perm_mem_win.adjust(n)
        if enable_retry:
            # Single RPC call: server-side colorize + auto-retry.
            img_color = colorizer.colorize_frame_with_retry(ti=n, frame_i=img_orig)
        else:
            img_color = colorizer.colorize_frame(ti=n, frame_i=img_orig)

        return img_to_frm(img_color, f[0].copy())

    if 0 < ref_weight < 1 and not (clip_sc is None):
        clip_colored = clip.std.ModifyFrame(clips=[clip, clip_ref, clip_sc],
                                            selector=partial(colormnet_client_color_merge, perm_mem_win=perm_mem_win,
                                                             reader=reader, colorizer=colorizer,
                                                             propagate=frame_propagate, weight=ref_weight))
    else:
        clip_colored = clip.std.ModifyFrame(clips=[clip, clip_ref],
                                            selector=partial(colormnet_client_color, perm_mem_win=perm_mem_win,
                                                             reader=reader, colorizer=colorizer,
                                                             enable_retry=enable_retry, propagate=frame_propagate))
    return clip_colored
