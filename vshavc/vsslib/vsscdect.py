"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2025-02-09
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Library of Vapoursynth utility functions.
"""

import vapoursynth as vs
import numpy as np
import cv2
from functools import partial
from skimage.metrics import structural_similarity

import vshavc.vsslib.vsutils as vsutil
import vshavc.vsslib.vsplugins as vsplugins

from vshavc.vsslib.constants import *
from vshavc.vsslib.vsplugins import load_MVTool_plugin
from vshavc.vsslib.vsresize import resize_min_HW

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
wrapper to function misc.SCDetect() (requires the dll: MiscFilters.dll)
if sc_tht_filter > 0 it will be activated the post change scene detection filter
based on SSIM. This metric is used to measure how similar are two images. 
It measures images luminance, contrast and structure and compare those values on 2 images.
Suggested values to use this features are:
threshold=0.05-0.09 (very sensitive threshold)
frequency=25
sc_tht_filter=0.65-0.75  
"""


def SceneDetect(clip: vs.VideoNode, threshold: float = DEF_THRESHOLD, frequency: int = 0, sc_tht_filter: float = 0,
                min_length: int = 1, tht_white: float = DEF_THT_WHITE, tht_black: float = DEF_THT_BLACK,
                frame_norm: bool = False, tht_offset: int = 1, sc_debug: bool = False) -> vs.VideoNode:
    """Detect scene changes and annotate frames with _SceneChangePrev/_SceneChangeNext properties.

    Stores the threshold and frequency parameters as frame properties ('sc_threshold',
    'sc_frequency'). For frequency==1 all frames are marked as scene changes. Otherwise
    delegates to SceneDetection (custom or misc.SCDetect) with optional SSIM post-filtering.

    :param clip:           Input clip (any format).
    :param threshold:      Luma-change threshold for scene detection [0, 1]. Default DEF_THRESHOLD.
    :param frequency:      If > 0, emit a scene change at least every 'frequency' frames. Default 0.
    :param sc_tht_filter:  SSIM threshold for post-detection refinement [0, 1]. 0 = disabled.
    :param min_length:     Minimum frame distance between consecutive scene changes. Default 1.
    :param tht_white:      Luma upper bound for scene changes (skip very bright frames). Default DEF_THT_WHITE.
    :param tht_black:      Luma lower bound for scene changes (skip very dark frames). Default DEF_THT_BLACK.
    :param frame_norm:     If True, normalise the GRAY8 clip before detection. Default False.
    :param tht_offset:     Frame comparison offset (>1 useful for blended scene changes). Default 1.
    :param sc_debug:       If True, log debug messages per frame. Default False.
    :return:               Clip with _SceneChangePrev/_SceneChangeNext frame properties set.
    """
    clip = clip.std.SetFrameProp(prop="sc_threshold", floatval=threshold)
    clip = clip.std.SetFrameProp(prop="sc_frequency", intval=frequency)

    if threshold == 0 and frequency == 0:
        return clip

    def set_scene_change_freq(n, f, freq: int = 1) -> vs.VideoFrame:

        f_out = f.copy()

        if freq == 1:
            f_out.props['_SceneChangePrev'] = 1
            f_out.props['_SceneChangeNext'] = 0
        elif n == 0:
            f_out.props['_SceneChangePrev'] = 1
            f_out.props['_SceneChangeNext'] = 0
        elif n % freq == 0:
            f_out.props['_SceneChangePrev'] = 1
            f_out.props['_SceneChangeNext'] = 0
        else:
            f_out.props['_SceneChangePrev'] = 0
            f_out.props['_SceneChangeNext'] = 0
        return f_out

    if frequency == 1 or (threshold == 0 and frequency > 1):
        return clip.std.ModifyFrame(clips=[clip], selector=partial(set_scene_change_freq, freq=frequency))

    try:
        sc_class = SceneDetection(sc_adaptive_ratio=DEF_ADAPTIVE_RATIO_MED if frequency > 0 else DEF_ADAPTIVE_RATIO_LO,
                                  sc_frequency=frequency,
                                  sc_tht_white=tht_white,
                                  sc_tht_black=tht_black,
                                  sc_debug=sc_debug)

        t_offset = min(max(tht_offset, 1), 25)
        m_length = min(max(min_length, 1), 25)
        sc = sc_class.SceneDetect(clip, threshold, sc_tht_filter, m_length, frame_norm, t_offset)
    except Exception as error:
        raise vs.Error("HAVC_colorizer: failure in SceneDetect(): -> " + str(error))

    return sc


def sc_clip_normalize(sc: vs.VideoNode, tht_white: float = DEF_THT_WHITE_MIN, tht_black: float = DEF_THT_BLACK_MIN
                      ) -> vs.VideoNode:
    """Normalise the luma range of a clip for frames in [tht_black, tht_white].

    Used before scene detection to increase sensitivity to smooth scene changes.

    :param sc:         Input GRAY/RGB clip.
    :param tht_white:  Upper luma bound for normalisation. Default DEF_THT_WHITE_MIN.
    :param tht_black:  Lower luma bound for normalisation. Default DEF_THT_BLACK_MIN.
    :return:           Normalised clip (unchanged for very dark/bright frames).
    """
    def set_normalize(n, f, tht_white: float, tht_black: float) -> vs.VideoFrame:
        frame_np = vsutil.frame_to_np_array(f)

        frame_m = vsutil.frame_normalize(frame_np, tht_black, tht_white)

        return vsutil.np_array_to_frame(frame_m, f.copy())

    sc = sc.std.ModifyFrame(clips=[sc], selector=partial(set_normalize, tht_white=tht_white, tht_black=tht_black))

    return sc


def get_sc_props(clip: vs.VideoNode) -> tuple[float, int]:
    """Read the 'sc_threshold' and 'sc_frequency' properties from the first frame of a clip.

    :param clip: Clip with 'sc_threshold' and 'sc_frequency' frame properties.
    :return:     Tuple (sc_threshold, sc_frequency); both 0 if properties are not found.
    """
    sc_threshold = 0
    sc_frequency = 0

    try:
        frame = clip.get_frame(0)
        sc_threshold = frame.props['sc_threshold']
        sc_frequency = frame.props['sc_frequency']
    except Exception as error:
        vs.core.log_message(2, "HAVC properties: 'sc_threshold', 'sc_frequency' not found in clip -> " + str(error))

    return sc_threshold, sc_frequency


def CopySCDetect(clip: vs.VideoNode, sc: vs.VideoNode) -> vs.VideoNode:
    """Copy scene-change and HAVC metadata frame properties from sc to clip.

    Copies: _SceneChangePrev, _SceneChangeNext, sc_threshold, sc_frequency, sc_luma, sc_ratio.

    :param clip: Destination clip.
    :param sc:   Source clip carrying the scene-change frame properties.
    :return:     clip with frame properties copied from sc.
    """
    return clip.std.CopyFrameProps(prop_src=sc, props=['_SceneChangePrev', '_SceneChangeNext',
                                                       'sc_threshold', 'sc_frequency', 'sc_luma', 'sc_ratio'])


def BuildSCDetect(clip_ref: vs.VideoNode) -> vs.VideoNode:
    """Create a blank clip with scene-change frame properties copied from clip_ref.

    Useful for propagating SC properties to a synthetic clip of the same length.

    :param clip_ref: Clip whose frame properties and dimensions define the output.
    :return:         Blank clip with _SceneChangePrev/_SceneChangeNext and HAVC props from clip_ref.
    """
    clip = vs.core.std.BlankClip(clip=clip_ref, length=clip_ref.num_frames,
                                 fpsnum=clip_ref.fps_num, fpsden=clip_ref.fps_den)
    return clip.std.CopyFrameProps(prop_src=clip_ref, props=['_SceneChangePrev', '_SceneChangeNext',
                                                             'sc_threshold', 'sc_frequency', 'sc_luma', 'sc_ratio'])


def SceneDetectFromDir(clip: vs.VideoNode, sc_framedir: str = None, merge_ref_frame: bool = False,
                       ref_frame_ext: bool = True) -> vs.VideoNode:
    """Mark frames as scene changes based on the reference image filenames in sc_framedir.

    A frame is flagged as a scene change when its index matches the number in a filename
    of the form ref_NNNNNN.ext. If merge_ref_frame is True, non-matching frames inherit
    their _SceneChangePrev/_SceneChangeNext from the input clip.

    :param clip:            Input clip.
    :param sc_framedir:     Directory containing reference image files (ref_NNNNNN.ext).
    :param merge_ref_frame: If True, non-reference frames keep their existing SC properties. Default False.
    :param ref_frame_ext:   If True, set _SceneChangeNext=1 on reference frames (marks them as external). Default True.
    :return:                Clip with _SceneChangePrev/_SceneChangeNext properties updated.
    """
    ref_list = vsutil.get_ref_names(sc_framedir)

    if len(ref_list) == 0:
        raise vs.Error(
            f"HAVC_deepex: no reference frames found in '{sc_framedir}', allowed format is: ref_nnnnnn.[png|jpg]")

    ref_num_list = [vsutil.get_ref_num(f) for f in ref_list]
    ref_num_list.sort()

    def set_scenechange(n: int, f: vs.VideoFrame, ref_num_list: list) -> vs.VideoFrame:

        fout = f.copy()

        if n in ref_num_list:
            fout.props['_SceneChangePrev'] = 1
            if ref_frame_ext:
                fout.props['_SceneChangeNext'] = 1
            else:
                fout.props['_SceneChangeNext'] = 0
        else:
            if merge_ref_frame:
                fout.props['_SceneChangePrev'] = f.props['_SceneChangePrev']
                fout.props['_SceneChangeNext'] = f.props['_SceneChangeNext']
            else:
                fout.props['_SceneChangePrev'] = 0
                fout.props['_SceneChangeNext'] = 0
        return fout

    sc = clip.std.ModifyFrame(clips=[clip], selector=partial(set_scenechange, ref_num_list=ref_num_list))

    return sc


class SceneDetection:
    """Stateful scene-change detector with adaptive ratio, luma filtering, and SSIM post-filtering.

    Maintains per-call state (last reference frame, previous diff, histogram) to implement
    an adaptive threshold that accounts for local content dynamics. Supports both the
    misc.SCDetect plugin path and a custom PlaneStats-based path.
    """

    _sc_debug: bool = None
    _sc_last_index = None
    _sc_last_ref = None
    _sc_prev_hist: np.ndarray = None
    _sc_prev_y = None
    _sc_prev_luma = None
    _sc_prev_diff = 0
    _sc_adaptive_ratio = None
    _sc_prev_index = None
    _sc_tht_white = None
    _sc_tht_black = None
    _sc_ref_luma = None
    _sc_frequency = 0

    def __init__(self, sc_adaptive_ratio: float = DEF_ADAPTIVE_RATIO_LO, sc_tht_white: float = DEF_THT_WHITE,
                 sc_tht_black: float = DEF_THT_BLACK, sc_frequency: int = 0, sc_debug: bool = False):
        """Initialise scene detection state.

        :param sc_adaptive_ratio: Ratio threshold for adaptive detection. Default DEF_ADAPTIVE_RATIO_LO.
        :param sc_tht_white:      Luma upper bound for valid scene changes. Default DEF_THT_WHITE.
        :param sc_tht_black:      Luma lower bound for valid scene changes. Default DEF_THT_BLACK.
        :param sc_frequency:      Minimum scene change frequency (frames). Default 0.
        :param sc_debug:          If True, log debug messages per frame. Default False.
        """
        self._sc_debug = sc_debug
        self._sc_last_index = None
        self._sc_last_ref = None
        self._sc_prev_y = None
        self._sc_prev_luma = None
        self._sc_ref_luma = None
        self._sc_prev_index = None
        self._sc_prev_diff = 0
        self._sc_prev_hist = None
        self._sc_adaptive_ratio = sc_adaptive_ratio
        self._sc_tht_white = sc_tht_white
        self._sc_tht_black = sc_tht_black
        self._sc_frequency = sc_frequency
        if self._sc_debug:
            vsutil.HAVC_LogMessage(vsutil.MessageType.WARNING,
                                   "sc_adaptive_ratio= ", sc_adaptive_ratio, ",  sc_tht_black= ", sc_tht_black,
                                   ", sc_tht_white= ", sc_tht_white, ", sc_frequency= ", sc_frequency)

    def SceneDetect(self, clip: vs.VideoNode, threshold: float = DEF_THRESHOLD, sc_tht_filter: float = 0,
                    min_length: int = 1, frame_norm: bool = False, tht_offset: int = 1) -> vs.VideoNode:
        """Run core scene detection on a clip.

        Routes to SceneDetectCustom (for fine thresholds, offsets > 1, or min_length > 1) or
        misc.SCDetect (standard path) followed by optional SSIM post-filtering and a
        luma-range filter (filter_black_white).

        :param clip:           Input clip (any format; internally converted to GRAY8).
        :param threshold:      Luma-change threshold [0, 1]. Default DEF_THRESHOLD.
        :param sc_tht_filter:  SSIM post-filter threshold [0, 1]. 0 = disabled.
        :param min_length:     Minimum frames between scene changes. Default 1.
        :param frame_norm:     Normalise luma before detection. Default False.
        :param tht_offset:     Frame comparison offset. Default 1.
        :return:               Clip with _SceneChangePrev/_SceneChangeNext properties set.
        """
        # add new properties for scene detection
        clip = clip.std.SetFrameProp(prop="sc_luma", floatval=0.5)
        clip = clip.std.SetFrameProp(prop="sc_ratio", floatval=0)

        sc = clip.resize.Bicubic(format=vs.GRAY8, matrix_s='709')
        sc = resize_min_HW(sc)
        try:
            if frame_norm:
                sc = sc_clip_normalize(sc)

            if sc_tht_filter > 0.0 or threshold < 0.10 or tht_offset > 1:
                if 0.0 < sc_tht_filter < 1.0 or min_length > 1:
                    sc = self.SceneDetectCustom(sc, threshold=threshold, offset=tht_offset)
                else:
                    sc = self.SceneDetectCustom(sc, threshold=threshold, offset=tht_offset,
                                                min_length=DEF_SC_MIN_DISTANCE)
            else:
                vsplugins.load_SCDetect_plugin()
                sc = sc.misc.SCDetect(threshold=threshold)
                sc = self.filter_black_white(clip, sc)

        except Exception as error:
            raise vs.Error("HAVC_colorizer: plugin 'MiscFilters.dll' not properly loaded/installed -> " + str(error))

        if 0.0 < sc_tht_filter < 1.0 or min_length > 1:
            clip_sc = resize_min_HW(clip)
            clip_sc = clip_sc.std.CopyFrameProps(prop_src=sc, props=['_SceneChangePrev', '_SceneChangeNext',
                                                               'sc_luma', 'sc_ratio'])
            clip_filter = self.SceneDetectFilter(clip=clip_sc, ssim_threshold=sc_tht_filter, min_length=min_length)
            clip_sc = clip.std.CopyFrameProps(prop_src=clip_filter, props=['_SceneChangePrev', '_SceneChangeNext',
                                                                  'sc_luma', 'sc_ratio'])
        else:
            clip_sc = clip.std.CopyFrameProps(prop_src=sc, props=['_SceneChangePrev', '_SceneChangeNext',
                                                                  'sc_luma', 'sc_ratio'])

        return clip_sc

    def filter_black_white(self, clip: vs.VideoNode, sc: vs.VideoNode) -> vs.VideoNode:
        """Filter out scene changes detected in too-dark or too-bright frames.

        Merges SC flags from sc into clip, suppressing them when frame luma is outside
        [tht_black, tht_white]. Also handles the frequency override.

        :param clip: Original input clip.
        :param sc:   Clip with raw _SceneChangePrev/_SceneChangeNext flags from misc.SCDetect.
        :return:     Clip with luma-filtered scene-change properties.
        """
        def set_scene_change(n, f, freq: int, tht_white: float, tht_black: float) -> vs.VideoFrame:

            f_out = f[0].copy()

            f_y = vsutil.frame_to_np_array(f[1])[:, :, 0]

            f_luma = round(np.mean(f_y) / 255.0, 4)
            # set luma property
            f_out.props['sc_luma'] = f_luma

            is_scenechange = (n == 0) or (f[1].props['_SceneChangePrev'] == 1 and f[1].props['_SceneChangeNext'] == 0)

            if freq > 1:
                is_scenechange = is_scenechange or (n % freq == 0)

            if is_scenechange and n == 0:
                # vs.core.log_message(2, "SceneDetect n= " + str(n))
                f_out.props['_SceneChangePrev'] = 1
                f_out.props['_SceneChangeNext'] = 0
            elif is_scenechange and tht_black < f_luma < tht_white:
                f_out.props['_SceneChangePrev'] = 1
                f_out.props['_SceneChangeNext'] = 0
            else:
                f_out.props['_SceneChangePrev'] = 0
                f_out.props['_SceneChangeNext'] = 0

            return f_out

        tht_white: float = self._sc_tht_white
        tht_black: float = self._sc_tht_black
        frequency: int = self._sc_frequency

        clip_new = clip.std.ModifyFrame(clips=[clip, sc], selector=partial(set_scene_change,
                                                                           freq=frequency,
                                                                           tht_white=tht_white,
                                                                           tht_black=tht_black))

        return clip_new

    def SceneDetectCustom(self, clip: vs.VideoNode, threshold: float = DEF_THRESHOLD, offset: int = 1,
                          min_length: int = 1) -> vs.VideoNode:
        """Custom scene detection using PlaneStats diff and an adaptive ratio threshold.

        Compares frame[n] with frame[n-offset] via PlaneStats. A scene change is triggered
        when the diff ratio exceeds sc_adaptive_ratio AND diff > threshold. Includes several
        override rules for frequency, very high ratios, and luma transitions.

        :param clip:        GRAY8 input clip.
        :param threshold:   Absolute luma-diff threshold [0, 1]. Default DEF_THRESHOLD.
        :param offset:      Frame comparison offset in [1, 25]. Default 1.
        :param min_length:  Minimum frame distance between consecutive detections. Default 1.
        :return:            GRAY8 clip with _SceneChangePrev/_SceneChangeNext properties set.
        """
        clip_prev = clip
        for i in range(offset):
            clip_prev = clip_prev.std.DuplicateFrames(frames=0).std.Trim(last=clip.num_frames - 1)
        clip_diff = vs.core.std.PlaneStats(clipa=clip_prev, clipb=clip, plane=0)

        def set_SCDetect(n, f, sc_threshold: float, min_length: int) -> vs.VideoFrame:

            f_out = f[0].copy()

            f_y = vsutil.frame_to_np_array(f_out)[:, :, 0]

            f_luma = round(np.mean(f_y) / 255.0, 4)
            f_luma_bright = DEF_THT_BLACK_MIN <= f_luma <= DEF_THT_WHITE_MIN
            # set luma property
            f_out.props['sc_luma'] = f_luma
            n_diff = round(max(float(f[1].props['PlaneStatsDiff']), 0.0001), 5)
            if n == 0 or self._sc_last_ref is None:
                is_scenechange = True
                self._sc_prev_diff = n_diff
                self._sc_ref_luma = f_luma
                self._sc_last_ref = n
                ratio = 0
            elif n - self._sc_last_ref < min_length:
                ratio = round(n_diff / self._sc_prev_diff, 4)
                is_scenechange = False
            else:
                ratio = round(n_diff / self._sc_prev_diff, 4)
                is_scenechange = ratio > self._sc_adaptive_ratio and n_diff > sc_threshold  # adaptive threshold
                self._sc_prev_diff = n_diff
                # override frequency
                if self._sc_frequency > 1:
                    is_scenechange = is_scenechange or (n % self._sc_frequency == 0)
                # override ratio
                is_scenechange = is_scenechange or (ratio > DEF_ADAPTIVE_RATIO_RF and f_luma_bright)
                is_scenechange = is_scenechange or ratio > DEF_ADAPTIVE_RATIO_VHI
                # override luma if previous luma is dark and current one is bright
                is_scenechange = is_scenechange or (self._sc_ref_luma < DEF_THT_BLACK_MIN and f_luma_bright)
                # final filtering on luma
                is_scenechange = is_scenechange and self._sc_tht_black < f_luma < self._sc_tht_white

            f_out.props['sc_ratio'] = ratio

            if self._sc_debug:
                vsutil.HAVC_LogMessage(vsutil.MessageType.WARNING,
                                       "Frame_n= ", n, ",  PlaneStatsDiff= ", n_diff,
                                       ", Ratio= ", ratio, ", PrvFrame= ", self._sc_prev_index,
                                       ", Luma= ", f_luma, ", SC=", is_scenechange)
            if is_scenechange:
                # vs.core.log_message(2, "SceneDetect n= " + str(n))
                self._sc_last_ref = n
                self._sc_ref_luma = f_luma
                f_out.props['_SceneChangePrev'] = 1
                f_out.props['_SceneChangeNext'] = 0
            else:
                f_out.props['_SceneChangePrev'] = 0
                f_out.props['_SceneChangeNext'] = 0

            self._sc_prev_index = n

            return f_out

        # sc = vsutil.debug_ModifyFrame(458, 467,
        #      clips=[clip, clip_diff], selector=partial(set_SCDetect, sc_threshold=threshold))

        sc = clip.std.ModifyFrame(clips=[clip, clip_diff], selector=partial(set_SCDetect, sc_threshold=threshold,
                                                                            min_length=min_length))

        return sc

    def SceneDetectFilter(self, clip: vs.VideoNode, ssim_threshold: float = 0.55, min_length: int = 1) -> vs.VideoNode:
        """Post-filter scene changes using SSIM and histogram similarity.

        Processes the clip in batches of 5000 frames to avoid memory pressure. For each
        candidate scene change, computes SSIM and Hellinger histogram distance against the
        previous scene-change frame; suppresses detections where images are too similar.

        :param clip:           Clip with candidate _SceneChangePrev flags (must include sc_luma/sc_ratio props).
        :param ssim_threshold: SSIM threshold below which a detection is accepted [0, 1]. Default 0.55.
        :param min_length:     Minimum frame distance between accepted scene changes. Default 1.
        :return:               Clip with refined _SceneChangePrev/_SceneChangeNext properties.
        """
        t_step = 5000  # batch size for the SSIM filter (to avoid buffer memory problems)
        clip_length = clip.num_frames

        clip_list = []

        for i in range(0, clip_length, t_step):
            t_start = i
            t_end = min(t_start + t_step, clip_length)
            clip_cut = clip[t_start:t_end]
            clip_i = self._scene_detect_filter_task(t_start, clip_cut, ssim_threshold, min_length)
            clip_list.append(clip_i)

        clip_sc = vs.core.std.Splice(clip_list)
        return clip_sc

    def _calc_histogram(self, y_img: np.ndarray, bins: int = 256, normalize: bool = True) -> np.ndarray:
        """Compute a (normalised) histogram of a grayscale image channel.

        :param y_img:     2-D uint8 grayscale image array.
        :param bins:      Number of histogram bins. Default 256.
        :param normalize: If True, normalise the histogram to [0, 1]. Default True.
        :return:          1-D float array of length bins.
        """
        # Extract Luma channel from the frame image

        # Create the histogram with a bin for every rgb value
        ht = cv2.calcHist([y_img], [0], None, [bins], [0, 256])
        if normalize:
            # Normalize the histogram
            ht = cv2.normalize(ht, ht).flatten()
        return ht

    def _scene_detect_filter_task(self, t_start: int, clip: vs.VideoNode, tht_ssim: float = 0.55, min_length: int = 1
                                  ) -> vs.VideoNode:
        """Process one batch of frames for SSIM/histogram post-filtering.

        Computes per-frame SSIM and Hellinger histogram distance relative to the last
        accepted scene change. Suppresses candidate detections where images are too similar.

        :param t_start:   Absolute frame offset of this batch (for correct global frame numbering).
        :param clip:      Batch clip with candidate _SceneChangePrev flags.
        :param tht_ssim:  SSIM threshold; below = scene change accepted. Default 0.55.
        :param min_length: Minimum frame distance between accepted scene changes. Default 1.
        :return:          Batch clip with refined _SceneChangePrev/_SceneChangeNext properties.
        """
        def set_scenechange(n: int, f: vs.VideoFrame, t_start: int, clip: vs.VideoNode, ssim_tht: float,
                            tht_white: float, tht_black, min_length: int = 1) -> vs.VideoFrame:
            fout = f.copy()
            luma: float = fout.props['sc_luma']
            ratio: float = fout.props['sc_ratio']
            np_frame = vsutil.frame_to_np_array(f)
            np_img = cv2.cvtColor(np_frame, cv2.COLOR_RGB2GRAY)
            y_img, _, _ = cv2.split(cv2.cvtColor(np_frame, cv2.COLOR_RGB2YUV))
            y_last = np_img
            t_n = t_start + n

            if t_n == 0:
                self._sc_last_index = None
                self._sc_prev_y = None
                self._sc_prev_hist = None
                self._sc_prev_luma = 0

            is_scenechange = fout.props['_SceneChangePrev'] == 1 or t_n == 0

            if is_scenechange and self._sc_last_index is None:
                fout.props['_SceneChangePrev'] = 1
                fout.props['_SceneChangeNext'] = 0
                self._sc_last_index = t_n
                self._sc_prev_y = y_last
                self._sc_prev_luma = luma
                self._sc_prev_hist = self._calc_histogram(y_img)
                if self._sc_debug:
                    vsutil.HAVC_LogMessage(vsutil.MessageType.WARNING,
                                           "SC=[New], Frame_n= ", t_n, ", PrvFrame= ", self._sc_last_index,
                                           ", SSIM= ", -1, ", Hist= ", -1, ", Luma= ", luma, ", ScReason= 1")
                return fout

            if not is_scenechange:
                return fout

            sc_reason = 0

            if is_scenechange and n > 0 and (t_n - self._sc_last_index) < min_length:
                if min_length > 1 and n > 1 and self._sc_prev_luma >= DEF_THT_BLACK_MIN > luma:
                    if self._sc_debug:
                        vsutil.HAVC_LogMessage(vsutil.MessageType.WARNING,
                                               "SC=[Skip], Frame_n= ", t_n, ", PrvFrame= ", self._sc_last_index,
                                               ", SSIM= ", -1, ", Hist = ", -1, ", Luma= ", luma, ", ScReason= -1")
                    fout.props['_SceneChangePrev'] = 0
                    fout.props['_SceneChangeNext'] = 0
                    return fout
                else:
                    sc_reason = 4

            y_hist = self._calc_histogram(y_img)

            if ssim_tht == 1:
                ssim_score = 1
                hist_score = 1
                scene_change = tht_black < luma < tht_white
                sc_reason = (sc_reason+1) if scene_change else 0
            elif n < clip.num_frames:
                ssim_score = round(structural_similarity(y_last, self._sc_prev_y, full=False), 4)
                hist_compare = cv2.compareHist(H1=self._sc_prev_hist, H2=y_hist, method=cv2.HISTCMP_HELLINGER)
                hist_score = round(1 - hist_compare, 4)
                if ssim_score < ssim_tht and hist_score < DEF_HIST_SCORE_HIGH:
                    scene_change = tht_black < luma < tht_white
                    # override on ratio and luma
                    if scene_change and sc_reason == 0 and self._sc_frequency > 1:
                        scene_change = (scene_change and
                                        not (luma < DEF_THT_BLACK_FREQ and ratio < DEF_ADAPTIVE_RATIO_RF))
                    sc_reason = (sc_reason+1) if scene_change else 0
                elif ssim_score >= DEF_SSIM_SCORE_EQUAL and self._sc_prev_luma < DEF_THT_BLACK_MIN <= luma:
                    # force scene change to get better frame
                    scene_change = tht_black < luma < tht_white
                    sc_reason = (sc_reason+2) if scene_change else 0
                elif ssim_score >= DEF_SSIM_SCORE_EQUAL and hist_score < DEF_HIST_SCORE_EQUAL:
                    scene_change = DEF_THT_BLACK_MIN < luma < DEF_THT_WHITE_MIN
                    sc_reason = (sc_reason+3) if scene_change else 0
                else:
                    scene_change = False
                    sc_reason = 0
            else:
                ssim_score = 1
                hist_score = 1
                scene_change = False

            if scene_change:
                if self._sc_debug:
                    vsutil.HAVC_LogMessage(vsutil.MessageType.WARNING,
                                           "SC=[New], Frame_n= ", t_n, ", PrvFrame= ", self._sc_last_index,
                                           ", SSIM= ", ssim_score, ", Hist= ", hist_score, ", Luma= ", luma,
                                           ", ScReason= ", sc_reason)
                fout.props['_SceneChangePrev'] = 1
                fout.props['_SceneChangeNext'] = 0
                self._sc_last_index = t_n
                self._sc_prev_y = y_last
                self._sc_prev_hist = y_hist
                self._sc_prev_luma = luma
            else:
                if self._sc_debug:
                    vsutil.HAVC_LogMessage(vsutil.MessageType.WARNING,
                                           "SC=[Skip], Frame_n: ", t_n, ", PrvFrame= ", self._sc_last_index,
                                           ", SSIM= ", ssim_score, ", Hist = ", hist_score, ", Luma= ", luma,
                                           ", ScReason= ", sc_reason)
                fout.props['_SceneChangePrev'] = 0
                fout.props['_SceneChangeNext'] = 0

            return fout

        # sc = vsutil.debug_ModifyFrame(45, 150, clips=[clip],
        #                              selector=partial(set_scenechange, t_start=t_start, clip=clip, ssim_tht=tht_ssim,
        #                                               tht_white=self._sc_tht_white, tht_black=self._sc_tht_black,
        #                                               min_length=min_length))

        sc = clip.std.ModifyFrame(clips=[clip],
                                  selector=partial(set_scenechange, t_start=t_start, clip=clip, ssim_tht=tht_ssim,
                                                   tht_white=self._sc_tht_white, tht_black=self._sc_tht_black,
                                                   min_length=min_length))

        return sc


def vs_sc_xvid(clip: vs.VideoNode, use_slices: bool = True) -> vs.VideoNode:
    """Detect scene changes using the SCXvid VapourSynth plugin.

    Converts the clip to YUV420P8 at reduced resolution for analysis, runs scxvid.Scxvid,
    and copies the resulting _SceneChangePrev/_SceneChangeNext properties back to the
    original clip.

    :param clip:       Input clip (any format).
    :param use_slices: Pass use_slices to scxvid.Scxvid. Default True.
    :return:           Original clip with _SceneChangePrev/_SceneChangeNext frame properties set.
    """
    vsplugins.load_SCXvid_plugin()

    sc_clip = resize_min_HW(clip, min_size = (480, 300))

    # adjusting color space YUV420P8 for vsSCXvidFilter
    sc_clip = vs.core.resize.Bicubic(clip=sc_clip, format=vs.YUV420P8, matrix_s="709", dither_type="error_diffusion")

    sc_clip = vs.core.scxvid.Scxvid(clip=sc_clip, use_slices=use_slices)

    # transfer '_SceneChangePrev', '_SceneChangeNext' to input clip
    clip = clip.std.CopyFrameProps(prop_src=sc_clip, props=['_SceneChangePrev', '_SceneChangeNext'])
    clip = clip.std.SetFrameProp(prop="sc_threshold", floatval=0.10)
    clip = clip.std.SetFrameProp(prop="sc_frequency", intval=0)

    return clip

def vs_mv_sc_detect(clip: vs.VideoNode, thscd1: int =250, thscd2: int =130, blksize: int = 16) -> vs.VideoNode:
    """
        Rileva i cambi scena usando i vettori di movimento.
        thscd1: Soglia SAD (Sum of Absolute Differences). Più bassa è, più è sensibile (default ~200-400).
        thscd2: Soglia blocchi. Più bassa è, più è sensibile (default ~100-150).

        Parametri su cui giocare:

        blksize: Usare blocchi più grandi (es. 16 o 32) rende l'analisi più veloce e meno suscettibile al rumore video.

        thscd1: Se hai molti "falsi positivi" (rileva cambi scena dove non ci sono), alza questo valore.
                Se non rileva i tagli netti, abbassalo.

        thscd2: Rappresenta quanti blocchi devono essere "mancanti" per dichiarare il cambio scena.
                Su un video 1080p con blocchi 16x16, ci sono circa 8100 blocchi. Un valore di 130 è piuttosto sensibile.

    """

    load_MVTool_plugin()

    low_res = resize_min_HW(clip, min_size=(480, 300))

    sc = low_res.resize.Bicubic(format=vs.GRAY8, matrix_s='709')

    # 1. Analisi del movimento (essenziale per SCDetection)
    #    Preparazione (Super Clip)
    #    Riduciamo la risoluzione per l'analisi (facoltativo, aumenta la velocità)
    #    'pel=1' è sufficiente per il cambio scena, non serve sub-pixel precision.
    sup = vs.core.mv.Super(sc, pel=1, sharp=0)

    # 2. Analisi dei vettori (Indietro nel tempo)
    #    isb=True (backward) analizza il frame precedente rispetto al corrente
    bvec = vs.core.mv.Analyse(sup, isb=False, blksize=blksize, overlap=8)

    # 3. Iniezione delle proprietà (_SceneChangePrev)
    sc = sc.std.SetFrameProp(prop="_SceneChangePrev", intval=0)
    sc = sc.std.SetFrameProp(prop="_SceneChangeNext", intval=0)

    sc_low_res = vs.core.mv.SCDetection(sc, bvec, thscd1=thscd1, thscd2=thscd2)

    # 4. Funzione per gestire la logica basata sulle proprietà
    """
    def handle_sc(n, f):
        is_scene_change = f.props.get('_SceneChangePrev', 0)

        # Qui puoi inserire la tua logica personalizzata.
        # Per ora, aggiungiamo un testo se richiesto.
        if show_text and is_scene_change:
            return vs.core.text.Text(clip, "CAMBIO SCENA RILEVATO", alignment=7)
        return clip

    # Usiamo FrameEval per valutare le proprietà frame per frame
    sc_output = vs.core.std.FrameEval(sc_low_res, handle_sc, prop_src=sc_low_res)
    """

    # transfer '_SceneChangePrev', '_SceneChangeNext' to input clip
    clip = clip.std.CopyFrameProps(prop_src=sc_low_res, props=['_SceneChangePrev', '_SceneChangeNext'])
    clip = clip.std.SetFrameProp(prop="sc_threshold", floatval=0.10)
    clip = clip.std.SetFrameProp(prop="sc_frequency", intval=0)

    return clip

