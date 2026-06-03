"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2026-05-19
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Library of Vapoursynth utility functions.
"""

import vapoursynth as vs
import os
import numpy as np
from PIL import Image
from enum import IntEnum
from functools import partial

IMG_EXTENSIONS = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.ppm', '.PPM', '.bmp', '.BMP']


class MessageType(IntEnum):
    """Enumeration of HAVC log message severity levels, mapped to VapourSynth message types."""

    DEBUG = vs.MESSAGE_TYPE_DEBUG,
    INFORMATION = vs.MESSAGE_TYPE_INFORMATION,
    WARNING = vs.MESSAGE_TYPE_WARNING,
    CRITICAL = vs.MESSAGE_TYPE_CRITICAL,
    FATAL = vs.MESSAGE_TYPE_FATAL  # also terminates the process, should generally not be used by normal filters
    EXCEPTION = 10  # raise a fatal exception that terminates the process


"""
def HAVC_LogMessage(message_type: MessageType = MessageType.INFORMATION, message_text: str = None):
    if message_type == MessageType.EXCEPTION:
        raise vs.Error(message_text)
    else:
        vs.core.log_message(int(message_type), message_text)
"""

def HAVC_LogMessage(message_type: MessageType = MessageType.INFORMATION, *args):
    """Log a message to the VapourSynth log or raise a fatal exception.

    When message_type is EXCEPTION, raises vs.Error (terminating the filter pipeline).
    All other types delegate to vs.core.log_message.

    :param message_type: Severity level from MessageType. Default INFORMATION.
    :param args:         Message parts; joined with spaces to form the message text.
    """
    message_text: str = ' '.join(map(str, args))
    if message_type == MessageType.EXCEPTION:
        raise vs.Error(message_text)
    else:
        vs.core.log_message(int(message_type), message_text)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function to convert a VideoFrame in Pillow image 
(why not available in Vapoursynth ?) 
"""


def frame_to_image(frame: vs.VideoFrame) -> Image:
    """Convert a VapourSynth VideoFrame (RGB24) to a PIL RGB image.

    :param frame: RGB24 VideoFrame to convert.
    :return:      PIL RGB Image with the same pixel data.
    """
    npArray = np.dstack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return Image.fromarray(npArray, 'RGB')


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function to convert a VideoFrame in Pillow image 
(why not available in Vapoursynth ?) 
"""


def frame_to_np_array(frame: vs.VideoFrame) -> np.ndarray:
    """Convert a VapourSynth VideoFrame (RGB24) to a NumPy array (H, W, 3), uint8.

    :param frame: RGB24 VideoFrame to convert.
    :return:      NumPy array with shape (H, W, 3).
    """
    npArray = np.dstack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return npArray


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function to convert a Pillow image in VideoFrame 
(why not available in Vapoursynth ?) 
"""


def image_to_frame(img: Image, frame: vs.VideoFrame) -> vs.VideoFrame:
    """Copy pixel data from a PIL RGB image into a VapourSynth VideoFrame.

    :param img:   PIL RGB image whose data to copy.
    :param frame: Target VideoFrame (must be writable, e.g. from f.copy()).
    :return:      The modified VideoFrame.
    """
    npArray = np.array(img)
    [np.copyto(np.asarray(frame[plane]), npArray[:, :, plane]) for plane in range(frame.format.num_planes)]
    return frame


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function to convert a np.array() image in VideoFrame 
"""


def np_array_to_frame(npArray: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
    """Copy pixel data from a NumPy array (H, W, 3) into a VapourSynth VideoFrame.

    :param npArray: NumPy array with shape (H, W, num_planes) to copy.
    :param frame:   Target VideoFrame (must be writable, e.g. from f.copy()).
    :return:        The modified VideoFrame.
    """
    [np.copyto(np.asarray(frame[plane]), npArray[:, :, plane]) for plane in range(frame.format.num_planes)]
    return frame


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Functions to save the reference frames of a clip 
"""

def _select_frames_by_list(clip: vs.VideoNode, frame_list: list[int]) -> vs.VideoNode:
    """
    Returns a new clip containing only the frames whose indices are in `frame_list`.
    The frames are output in the order they appear in `frame_list`.
    Frame indices must be within [0, clip.num_frames].
    """
    if not frame_list:
        raise ValueError("frame_list is empty")

    # Validate frame numbers
    max_frame = clip.num_frames - 1
    if any(n < 0 or n > max_frame for n in frame_list):
        raise ValueError("Frame numbers in list must be in range [0, clip.num_frames - 1]")

    # Create a list of single-frame clips
    selected_clips = [clip[n] for n in frame_list]

    # Splice them into a single clip
    return vs.core.std.Splice(selected_clips)

# global variable for sc counting
_sc_counter: int
_sc_list: list[int]


def vs_sc_export_frames(clip: vs.VideoNode = None, sc_framedir: str = None, ref_offset: int = 0,
                        ref_ext: str = 'png', ref_jpg_quality: int = 95, ref_override: bool = True,
                        prop_name: str = "_SceneChangePrev", sequence: bool = False) -> vs.VideoNode:
    """Export scene-change frames to a directory as image files.

    Frames flagged by prop_name are saved to sc_framedir as ref_NNNNNN.ext. When
    sequence=True the filename counter increments for each exported frame regardless of
    the actual frame number; otherwise the frame number (plus ref_offset) is used.

    :param clip:            RGB24 input clip.
    :param sc_framedir:     Output directory for exported frames.
    :param ref_offset:      Added to the frame number in the filename. Default 0.
    :param ref_ext:         Image format extension ('png' or 'jpg'). Default 'png'.
    :param ref_jpg_quality: JPEG quality when ref_ext='jpg'. Default 95.
    :param ref_override:    If False, skip frames whose file already exists. Default True.
    :param prop_name:       Frame property used to detect scene changes. Default '_SceneChangePrev'.
    :param sequence:        If True, use a sequential counter instead of frame numbers. Default False.
    :return:                Clip pass-through (side-effect: frames saved to disk).
    """
    pil_ext = ref_ext.lower()
    global _sc_counter
    _sc_counter = 0

    def save_sc_frame(n, f, sc_framedir: str = None, ref_offset: int = 0, prop_name: str = "_SceneChangePrev",
                      ref_ext: str = 'png', ref_jpg_quality: int = 95, ref_override: bool = True,
                      sequence: bool = False):
        global _sc_counter

        is_scenechange = (n == 0) or (f.props[prop_name] == 1)
        if is_scenechange:
            if sequence:
                ref_n = _sc_counter
                _sc_counter = _sc_counter + 1
            else:
                ref_n = n + ref_offset
            img = frame_to_image(f)
            img_path = os.path.join(sc_framedir, f"ref_{ref_n:06d}.{ref_ext}")
            if not ref_override and os.path.exists(img_path):
                return f.copy()  # do nothing
            if ref_ext == "jpg":
                img.save(img_path, subsampling=0, quality=ref_jpg_quality)
            else:
                img.save(img_path)

        return f.copy()

    clip = clip.std.ModifyFrame(clips=[clip], selector=partial(save_sc_frame, sc_framedir=sc_framedir,
                                                               ref_offset=ref_offset, prop_name=prop_name,
                                                               ref_ext=pil_ext, ref_jpg_quality=ref_jpg_quality,
                                                               ref_override=ref_override, sequence=sequence))

    return clip


def vs_list_export_frames(clip: vs.VideoNode = None, sc_framedir: str = None, ref_list: list[int] = None,
                          offset: int = 0, ref_ext: str = 'png', ref_jpg_quality: int = 95, ref_override: bool = True,
                          fast_extract: bool = True) -> vs.VideoNode:
    """Export frames from an explicit list of frame indices to a directory.

    When fast_extract=True, only the listed frames are extracted (via _select_frames_by_list)
    before saving, which is faster than iterating the whole clip. A single-element ref_list
    is treated as a step value and automatically expanded to range(0, num_frames, ref_list[0]).

    :param clip:            RGB24 input clip.
    :param sc_framedir:     Output directory for exported frames.
    :param ref_list:        List of frame indices to export, or a single-element step list.
    :param offset:          Offset added to each frame index in the filename. Default 0.
    :param ref_ext:         Image format extension ('png' or 'jpg'). Default 'png'.
    :param ref_jpg_quality: JPEG quality when ref_ext='jpg'. Default 95.
    :param ref_override:    If False, skip frames whose file already exists. Default True.
    :param fast_extract:    If True, pre-filter the clip to the listed frames (faster). Default True.
    :return:                Clip pass-through (side-effect: frames saved to disk).
    """
    pil_ext = ref_ext.lower()

    if len(ref_list) == 1: # the list is automatically generated
        sorted_list = list(range(0, clip.num_frames, ref_list[0]))
    else: # the list is sorted and duplicate frames are removed
        sorted_list = sorted(set(ref_list))

    if offset > 0:
        sorted_list = [num + offset for num in sorted_list]
    
    def save_sc_frame(n, f, sc_framedir: str = None, ref_list: list[int] = None, ref_ext: str = 'png',
                      ref_jpg_quality: int = 95, ref_override: bool = True, fast_extract: bool = True):
        global _sc_counter

        if fast_extract:
            is_scenechange = True
            f_num = ref_list[n]
        else:
            is_scenechange = (n in ref_list)
            f_num = n
        if is_scenechange:
            img = frame_to_image(f)
            img_path = os.path.join(sc_framedir, f"ref_{f_num:06d}.{ref_ext}")
            if not ref_override and os.path.exists(img_path):
                return f.copy()  # do nothing
            if ref_ext == "jpg":
                img.save(img_path, subsampling=0, quality=ref_jpg_quality)
            else:
                img.save(img_path)

        return f.copy()

    if fast_extract:
        clip_ref = _select_frames_by_list(clip, sorted_list)
    else:
        clip_ref = clip

    clip_new = clip_ref.std.ModifyFrame(clips=[clip_ref], selector=partial(save_sc_frame, sc_framedir=sc_framedir,
                                        ref_list=sorted_list, ref_ext=pil_ext, ref_jpg_quality=ref_jpg_quality,
                                        ref_override=ref_override, fast_extract=fast_extract))

    #clip_new = debug_ModifyFrame(f_start=0, f_end=147, clip=clip_ref, clips=[clip_ref],
    #                             selector=partial(save_sc_frame, sc_framedir=sc_framedir,
    #                                    ref_list=sorted_list, ref_ext=pil_ext, ref_jpg_quality=ref_jpg_quality,
    #                                    ref_override=ref_override, fast_extract=fast_extract), silent=True)
    return clip_new


def vs_get_video_ref(clip: vs.VideoNode = None, prop_name: str = "_SceneChangePrev") -> vs.VideoNode:
    """Annotate each frame with a 'sc_next_frame' property pointing to the next scene-change frame.

    First pass: collects all scene-change frame numbers into _sc_list. Second pass: sets the
    'sc_next_frame' property to the next scene-change frame number at each scene-change frame,
    and 0 for non-scene-change frames; -1 signals the end of the list.

    :param clip:      RGB24 input clip.
    :param prop_name: Frame property used to detect scene changes. Default '_SceneChangePrev'.
    :return:          Clip with 'sc_next_frame' frame property set per frame.
    """
    global _sc_list, _sc_counter
    _sc_list = []

    def get_sc_list(n, f, prop_name: str):
        global _sc_list

        is_scenechange = (n == 0) or (f.props[prop_name] == 1)
        if is_scenechange:
            _sc_list.append(n)
        return f.copy()

    clip = clip.std.ModifyFrame(clips=[clip], selector=partial(get_sc_list, prop_name=prop_name))

    # set property to set the next reference frame position
    clip = clip.std.SetFrameProp(prop="sc_next_frame", intval=0)
    _sc_counter = 0

    def set_sc_list(n, f, sc_list: list[int], prop_name: str):
        global _sc_counter
        f_out = f.copy()
        is_scenechange = (n == 0) or (f.props[prop_name] == 1)
        if is_scenechange:
            if _sc_counter < len(sc_list):
                f_out.props["sc_next_frame"] = sc_list[_sc_counter]
            else:
                f_out.props["sc_next_frame"] = -1   # end list
            _sc_counter = _sc_counter + 1
        else:
            f_out.props["sc_next_frame"] = 0

        return f.copy()

    clip = clip.std.ModifyFrame(clips=[clip], selector=partial(set_sc_list, sc_list=_sc_list, prop_name=prop_name))

    return clip


def get_ref_last_list() -> list[int]:
    """Return the global list of scene-change frame numbers collected by vs_get_video_ref."""
    global _sc_list
    return _sc_list


def get_ref_num(filename: str = ""):
    """Extract the frame number from a reference filename (format: ref_NNNNNN.ext).

    :param filename: Reference filename string.
    :return:         Integer frame number.
    """
    fname = filename.split(".")[0]
    fnum = int(fname.split("_")[-1])
    return fnum


def get_ref_images(in_dir="./") -> list:
    """Return a list of full paths to reference image files in in_dir.

    Only files matching the ref_NNNNNN naming convention and a supported extension
    (as determined by is_ref_file) are included.

    :param in_dir: Directory to scan. Default './'.
    :return:       List of absolute file paths.
    """
    img_ref_file = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if is_ref_file(in_dir, f)]
    return img_ref_file


def get_ref_names(in_dir="./") -> list:
    """Return a list of filenames (not full paths) of reference images in in_dir.

    :param in_dir: Directory to scan. Default './'.
    :return:       List of filenames.
    """
    img_ref_list = [f for f in os.listdir(in_dir) if is_ref_file(in_dir, f)]
    return img_ref_list


def is_ref_file(in_dir="./", fname: str = "") -> bool:
    """Return True if fname is a valid reference image file (starts with 'ref_', supported extension).

    :param in_dir: Directory containing the file.
    :param fname:  Filename to check.
    :return:       True if the file exists and matches the reference naming convention.
    """
    filename = os.path.join(in_dir, fname)

    if not os.path.isfile(filename):
        return False

    return fname.startswith("ref_") and any(fname.endswith(extension) for extension in IMG_EXTENSIONS)


def frame_normalize(frame_np: np.ndarray, tht_black: float = 0.10, tht_white: float = 0.90) -> np.ndarray:
    """Normalise the Y (luma) plane of a frame to [0, 255] when its average luma is in [tht_black, tht_white].

    Frames that are too dark or too bright are returned unchanged to avoid over-normalisation.

    :param frame_np:  Input array (H, W, 3), uint8, with Y in plane 0.
    :param tht_black: Minimum average luma for normalisation to be applied. Default 0.10.
    :param tht_white: Maximum average luma for normalisation to be applied. Default 0.90.
    :return:          Normalised array (or original if outside the luma range).
    """
    frame_y = frame_np[:, :, 0]

    frame_luma = np.mean(frame_y) / 255.0

    if frame_luma <= tht_black or frame_luma >= tht_white:
        return frame_np

    img_np = frame_np.copy()

    frame_y = np.multiply(255, (frame_y - np.min(frame_y)) / (np.max(frame_y) - np.min(frame_y)))

    img_np[:, :, 0] = frame_y.clip(0, 255).astype('uint8')

    return img_np


def mean_pixel_distance(y_left: np.ndarray, y_right: np.ndarray, normalize: bool = True) -> float:
    """Return the mean average distance in pixel values between `left` and `right`.
    Both `left and `right` should be 2-dimensional 8-bit images of the same shape.
    """

    if normalize:
        luma_left = int(np.mean(y_left))
        luma_right = int(np.mean(y_right))
        if luma_right > luma_left:
            y_left = (y_left + (luma_right - luma_left)).clip(0, 255).astype('uint8')
        else:
            y_right = (y_right - (luma_right - luma_left)).clip(0, 255).astype('uint8')

    num_pixels: float = float(y_left.shape[0] * y_left.shape[1])
    dist = np.sum(np.abs(y_left.astype(np.int32) - y_right.astype(np.int32))) / num_pixels
    return dist / 255.0

def SCDetect(clip: vs.VideoNode, threshold: float = 0.1, plane: int = 0) -> vs.VideoNode:
    """
    Scene change detection with _SceneChangePrev/_SceneChangeNext frame properties.
    Uses core.misc.SCDetect if available (plane=0 only), otherwise falls back to
    a std.PlaneStats-based reimplementation.

    Args:
        clip      : Input clip
        threshold : Scene change threshold (default: 0.1, must be 0.0–1.0)
        plane     : Plane to analyze; only honoured in fallback path —
                    misc.SCDetect always uses plane 0

    Returns:
        Clip with _SceneChangePrev and _SceneChangeNext frame properties set.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('SCDetect: this is not a clip')
    if not (0.0 <= threshold <= 1.0):
        raise vs.Error('SCDetect: threshold must be between 0.0 and 1.0')
    if clip.num_frames < 2:
        raise vs.Error('SCDetect: clip must have more than one frame')

    if hasattr(vs.core, 'misc') and plane == 0:
        if clip.format.color_family == vs.RGB:
            if clip.format != vs.GRAY8:
                sc = clip.resize.Point(format=vs.GRAY8, matrix_s='709')
            sc = vs.core.misc.SCDetect(sc, threshold=threshold)

            def _copy_props(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
                fout = f[0].copy()
                fout.props['_SceneChangePrev'] = f[1].props['_SceneChangePrev']
                fout.props['_SceneChangeNext'] = f[1].props['_SceneChangeNext']
                return fout

            return clip.std.ModifyFrame(clips=[clip, sc], selector=_copy_props)

        return vs.core.misc.SCDetect(clip, threshold=threshold)

    # prev_stats[n] = diff(frame_{n-1}, frame_n) → SceneChangePrev
    # next_stats[n] = diff(frame_n, frame_{n+1}) → SceneChangeNext
    prev_shifted = clip.std.DuplicateFrames(0).std.Trim(last=clip.num_frames - 1)
    prev_stats = vs.core.std.PlaneStats(prev_shifted, clip, plane=plane)
    next_shifted = clip.std.DuplicateFrames(clip.num_frames - 1).std.Trim(first=1)
    next_stats = vs.core.std.PlaneStats(clip, next_shifted, plane=plane)

    def _set_sc_props(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        fout = f[0].copy()
        fout.props['_SceneChangePrev'] = int(float(f[1].props.get('PlaneStatsDiff', 0.0)) > threshold)
        fout.props['_SceneChangeNext'] = int(float(f[2].props.get('PlaneStatsDiff', 0.0)) > threshold)
        return fout

    return clip.std.ModifyFrame(
        clips=[clip, prev_stats, next_stats],
        selector=_set_sc_props
    )

def debug_ModifyFrame(f_start: int = 0, f_end: int = 1, clip: vs.VideoNode = None,
                      clips: list[vs.VideoNode] = None, selector: partial = None, silent: bool = True) -> vs.VideoNode:
    """Debug helper: manually execute a ModifyFrame selector over a range of frames.

    Calls selector(n, frame) for each frame in [f_start, f_end) without building a
    VapourSynth pipeline, which makes it useful for inspecting per-frame logic in Python.
    Returns the original clip unchanged.

    :param f_start:  First frame to process. Default 0.
    :param f_end:    Last frame (exclusive). Clamped to clip length. Default 1.
    :param clip:     Clip whose length defines the valid frame range.
    :param clips:    List of input clips passed to selector (1 or more).
    :param selector: Callable with signature (n, f) or (n, [f0, f1, ...]).
    :param silent:   If False, print the frame number before each call. Default True.
    :return:         Original clip (pass-through; side-effects from selector apply).
    """
    f_end = min(f_end, clip.num_frames - 1)
    if len(clips) == 1:
        if f_start > 0:
            frame = clips[0].get_frame(0)
            if not silent:
                print("Debug Frame: ", 0)
            selector(0, frame)
        for n in range(f_start, f_end):
            frame = clips[0].get_frame(n)
            if not silent:
                print("Debug Frame: ", n)
            selector(n, frame)
    else:
        if f_start > 0:
            frame = []
            for j in range(0, len(clips)):
                frame.append(clips[j].get_frame(0))
            if not silent:
                print("Debug Frame: ", 0)
            selector(0, frame)
        for n in range(f_start, f_end):
            frame = []
            for j in range(0, len(clips)):
                frame.append(clips[j].get_frame(n))
            if not silent:
                print("Debug Frame: ", n)
            selector(n, frame)

    return clip

def debug_FrameEval(f_start: int = 0, f_end: int = 1, clip: vs.VideoNode = None,
                      eval: partial = None, silent: bool = True) -> vs.VideoNode:
    """Debug helper: manually execute a FrameEval callback over a range of frames.

    Calls eval(n) for each frame in [f_start, f_end) outside the VapourSynth pipeline.
    Returns the original clip unchanged.

    :param f_start: First frame to evaluate. Default 0.
    :param f_end:   Last frame (exclusive). Clamped to clip length. Default 1.
    :param clip:    Clip whose length defines the valid frame range.
    :param eval:    Callable with signature (n,).
    :param silent:  If False, print the frame number before each call. Default True.
    :return:        Original clip (pass-through).
    """
    f_end = min(f_end, clip.num_frames - 1)
    for n in range(f_start, f_end):
        if not silent:
            print("Debug Frame: ", n)
        eval(n)

    return clip