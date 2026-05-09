"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-04-08
version:
LastEditors: Dan64
LastEditTime: 2026-01-15
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Library of Vapoursynth utility functions for edge based scene detection.
"""

import vapoursynth as vs
from vapoursynth import core
from typing import Optional
from functools import partial

import numpy as np
import cv2
from functools import partial
from skimage.metrics import structural_similarity
import vsdeoldify.vsslib.vsplugins as vsplugins
from vsdeoldify.vsslib.constants import *
import vsdeoldify.vsslib.vsutils as vsutil
from vsdeoldify.vsslib.constants import DEF_THT_WHITE, DEF_THT_BLACK
from vsdeoldify.vsslib.vsplugins import load_Retinex_plugin, load_TCanny_plugin, load_Akarin_plugin, load_SCDetect_plugin
from vsdeoldify.vsslib.vsresize import resize_min_HW
from vsdeoldify.vsslib.vsutils import frame_to_image


def SceneDetectEdges(clip: vs.VideoNode, threshold: float = 0.07, frequency: int = 0, ssim_threshold: float = 0.0,
                     sc_diff_offset: int = 2, sc_min_int:int = 30, sc_mult_tht: int = 7, tht_white: float =0.70,
                     tht_black: float =0.12, sc_debug: bool = False) -> vs.VideoNode:
    """Edge-based scene change detector using a Retinex-enhanced Kirsch/TCanny edge mask.

    Computes masked pixel-difference statistics between temporally offset frames, combining
    them with a misc.SCDetect pre-pass. Optional SSIM post-filtering (SceneDetectionFiltered)
    further refines the detections. Annotates _SceneChangePrev/_SceneChangeNext frame properties.

    :param clip:            Input clip (any format).
    :param threshold:       Edge-diff threshold for scene detection [0, 1]. Default 0.07.
    :param frequency:       If > 0, emit a scene change at least every 'frequency' frames. Default 0.
    :param ssim_threshold:  SSIM post-filter threshold [0, 1]. 0 = disabled. Default 0.0.
    :param sc_diff_offset:  Frame comparison offset in frames (≥ 1). Default 2.
    :param sc_min_int:      Minimum frame distance between scene changes. Default 30.
    :param sc_mult_tht:     Multiplier for the mandatory high-threshold override. Default 7.
    :param tht_white:       Luma upper bound for valid scene changes. Default 0.70.
    :param tht_black:       Luma lower bound for valid scene changes. Default 0.12.
    :param sc_debug:        If True, log per-frame debug messages. Default False.
    :return:                Clip with _SceneChangePrev/_SceneChangeNext properties set.
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

    sc_mult_tht = 7 if sc_mult_tht == 0 else sc_mult_tht
    sc_diff_offset=max(sc_diff_offset, 1)

    try:

        # add new properties for scene detection
        clip = clip.std.SetFrameProp(prop="sc_luma", floatval=0.15)
        clip = clip.std.SetFrameProp(prop="sc_reason", floatval=0)

        edge_threshold = round(1.75*threshold,5)

        sc = vs_edge_based_scenedetect(
            clip,
            ssim_diff_threshold=edge_threshold,
            edge_diff_threshold=threshold,
            sc_diff_offset=sc_diff_offset,
            sc_min_distance=sc_min_int,
            sc_mult_tht=sc_mult_tht,
            tht_white=tht_white,
            tht_black=tht_black,
            sc_debug=sc_debug
        )

        if ssim_threshold > 0:
            min_length = max(int(round(sc_min_int/3.0)),1)
            sc_class = SceneDetectionFiltered(sc_tht_white=tht_white, sc_tht_black=tht_black, sc_frequency=frequency,
                                              sc_debug=sc_debug)
            clip_small = resize_min_HW(clip)
            clip_small = clip_small.std.CopyFrameProps(prop_src=sc, props=['_SceneChangePrev',
                                                                           '_SceneChangeNext',
                                                                           'sc_luma'
                                                                           'sc_reason'])
            sc_filter = sc_class.SceneDetectFilter(clip=clip_small, ssim_threshold=ssim_threshold, min_length=min_length)
            sc = clip.std.CopyFrameProps(prop_src=sc_filter, props=['_SceneChangePrev', '_SceneChangeNext',
                                                                    'sc_luma', 'sc_reason'])

    except Exception as error:
        raise vs.Error("HAVC_colorizer: failure in SceneDetect(): -> " + str(error))

    return sc

def TemporalMedian(clip, radius=1, planes=None):
    """Temporal median filter using zsmooth.TemporalMedian.

    :param clip:   Input clip.
    :param radius: Temporal radius. Default 1.
    :param planes: Plane list to filter. None = all planes.
    :return:       Temporally median-filtered clip.
    """
    return core.zsmooth.TemporalMedian(clip, radius=radius, planes=planes)

def Median(clip, radius=1, planes=None):
    """Spatial median filter with fallback: tries zsmooth.Median, then std.Median (radius==1 only).

    :param clip:   Input clip.
    :param radius: Filter radius. Default 1.
    :param planes: Plane list to filter. None = all planes.
    :return:       Spatially median-filtered clip.
    """
    # fallback plugin because zsmooth does not support non AVX2 CPUs. use std.Median for r=1 and CTMF for higher.
    if hasattr(core, "zsmooth"):
        return core.zsmooth.Median(clip, radius=radius, planes=planes)
    elif radius == 1:
        return core.std.Median(clip, planes=planes)

def kirsch(src: vs.VideoNode) -> vs.VideoNode:
    """Compute a Kirsch edge-detection map via 4 directional 3×3 convolutions.

    Each of the 4 rotated kernels detects edges in a different direction; the final
    result is the per-pixel maximum across all 4 directions (via akarin.Expr).

    :param src: GRAY8 or single-plane input clip.
    :return:    Edge-strength clip (same format as src).
    """
    w = [5]*3 + [-3]*5
    weights = [w[-i:] + w[:-i] for i in range(4)]
    c = [src.std.Convolution((w[:4]+[0]+w[4:]), saturate=False) for w in weights]
    return core.akarin.Expr(c, 'x y max z max a max')

def retinex_edgemask(rgb: vs.VideoNode, sigma: float = 1.0, draft: bool = False) -> vs.VideoNode:
    """Build an edge mask combining Retinex/gamma enhancement, Kirsch, and TCanny.

    In draft mode a fast gamma-sqrt enhancement is used instead of Retinex MSRCP.
    The final mask is the per-pixel sum of Kirsch and TCanny edges, clamped to [0, 255].

    :param rgb:    GRAY8 input clip.
    :param sigma:  TCanny Gaussian sigma. Default 1.0.
    :param draft:  If True, use fast gamma boost instead of MSRCP Retinex. Default False.
    :return:       GRAY8 edge-strength clip.
    """
    if draft:
        # Gamma boost: sqrt(x/255) * 255
        #enhanced = core.std.Expr(rgb, 'x 255 / sqrt 255 *')
        enhanced = core.akarin.Expr(rgb, 'x 255 / sqrt 255 *')
    else:
        msr = core.retinex.MSRCP(rgb, sigma=[50, 200, 350], upper_thr=0.005)
        enhanced = core.std.ShufflePlanes(msr, 0, vs.GRAY)
        enhanced = core.std.Limiter(enhanced, 0, 255)

    kirsch_edge = kirsch(rgb)

    tcanny_edge = enhanced.tcanny.TCanny(mode=1, sigma=sigma)

    return core.std.Expr([kirsch_edge, tcanny_edge], 'x y + 255 min')

# Variabile globale
_last_sc_frame = -1000  # inizializza fuori dalla funzione
_last_sc_status = ""

def vs_edge_based_scenedetect(
    clip: vs.VideoNode,
    ssim_diff_threshold: float = 0.10,
    edge_diff_threshold: float = 0.07,
    sc_diff_offset: int = 2,
    sc_min_distance: int = 10,
    sc_mult_tht: int = 7,
    tht_white: float =0.80,
    tht_black: float =0.10,
    canny_sigma: float = 1.2,
    sc_debug: bool = False
) -> vs.VideoNode:
    """Core edge-based scene detection using a Retinex-enhanced edge mask and PlaneStats.

    Computes masked (edge-weighted) pixel-difference statistics between frame[n] and
    frame[n+sc_diff_offset]. A scene change is accepted when both the SSIM-style total
    diff and the edge-masked diff exceed their respective thresholds, subject to the
    sc_min_distance enforcement and sc_mult_tht override for very large edges.
    Very dark or very bright frames (outside [tht_black, tht_white]) are suppressed.

    :param clip:                Input clip (any format).
    :param ssim_diff_threshold: Total pixel-diff threshold (4× PlaneStatsAverage). Default 0.10.
    :param edge_diff_threshold: Masked edge-diff threshold (10× PlaneStatsAverage). Default 0.07.
    :param sc_diff_offset:      Temporal offset between compared frames (≥ 1). Default 2.
    :param sc_min_distance:     Minimum frame distance between scene changes. Default 10.
    :param sc_mult_tht:         Multiplier for mandatory high-edge override. Default 7.
    :param tht_white:           Luma upper bound for valid scene changes. Default 0.80.
    :param tht_black:           Luma lower bound for valid scene changes. Default 0.10.
    :param canny_sigma:         TCanny sigma for the edge mask. Default 1.2.
    :param sc_debug:            If True, log per-frame debug messages. Default False.
    :return:                    Clip with _SceneChangePrev/_SceneChangeNext properties set.
    """
    global _last_sc_frame, _last_sc_status
    _last_sc_frame = -sc_min_distance  # resetta ad ogni chiamata
    _last_sc_status = ""

    # Caricamento plugins
    load_SCDetect_plugin()
    load_Retinex_plugin()
    load_TCanny_plugin()
    load_Akarin_plugin()

    # --- 1. Preparazione GRAY16 ---
    gray = core.resize.Bicubic(clip, format=vs.GRAY8, matrix_s="709")
    gray = resize_min_HW(gray)
    sc_gray = gray.misc.SCDetect(threshold=0.10)

    clip_curr = gray
    clip_next = gray[sc_diff_offset:] + gray[-sc_diff_offset]

    # --- 2. Maschera edge + differenza ---
    edge_mask = retinex_edgemask(gray, sigma=canny_sigma, draft=True)
    #diff = core.std.Expr([clip_curr, clip_next], 'x y - abs')
    diff = core.akarin.Expr([clip_curr, clip_next], 'x y - abs')
    masked_diff = core.std.MaskedMerge(core.std.BlankClip(diff), diff, edge_mask)
    maskdiff_stats = core.std.PlaneStats(masked_diff)
    diff_stats = core.std.PlaneStats(diff)

    def set_sc_prop(n, f, ssim_diff_threshold:float, edge_diff_threshold: float, diff_stats: vs.VideoNode,
                    maskdiff_stats: vs.VideoNode, sc_min_distance: int, sc_mult_tht:int , tht_white: float,
                    tht_black: float, sc_debug) -> vs.VideoNode:

        global _last_sc_frame, _last_sc_status
        f_out = f[0].copy()

        if n == 0:
            f_out.props['_SceneChangePrev'] = 1
            f_out.props['_SceneChangeNext'] = 0
            f_out.props['sc_luma'] = 0.10
            f_out.props['sc_reason'] = 4
            _last_sc_frame = 0
            _last_sc_status = "Accepted(First)"
            return f_out

        # --------------------
        f_y = vsutil.frame_to_np_array(f[1])[:, :, 0]
        f_luma: float = round(np.mean(f_y) / 255.0, 4)

        edge_diff = round(10*maskdiff_stats.get_frame(n).props.PlaneStatsAverage, 5)
        ssim_diff = round(4*diff_stats.get_frame(n).props.PlaneStatsAverage, 5)

        # Resetta SC se fuori range di luma
        in_luma_range = tht_black <= f_luma <= tht_white
        above_threshold = (edge_diff > edge_diff_threshold) and (ssim_diff > ssim_diff_threshold)
        above_distance_max = (n - _last_sc_frame) >= sc_min_distance
        above_distance_min = (n - _last_sc_frame) >= max(int(sc_mult_tht*0.5), 3)
        mandatory_ref_1 = f[1].props['_SceneChangePrev'] == 1
        mandatory_ref_2 = edge_diff > (edge_diff_threshold*sc_mult_tht)

        f_out.props['sc_luma'] = f_luma
        f_out.props['sc_reason'] = 0
        f_out.props['_SceneChangePrev'] = 0
        f_out.props['_SceneChangeNext'] = 0

        if in_luma_range:
            if mandatory_ref_1:
                if ("tht_max" not in _last_sc_status) or above_distance_min:
                    f_out.props['_SceneChangePrev'] = 1
                    _last_sc_frame = n
                    if mandatory_ref_2:
                        status = "Accepted(tht_max+edge_max)"
                        f_out.props['sc_reason'] = 4
                    else:
                        status = "Accepted(tht_max)"
                        f_out.props['sc_reason'] = 3
                    _last_sc_status = status
                else:
                    status = "Skipped"
            elif mandatory_ref_2:
                if ("edge_max" not in _last_sc_status) or above_distance_min:
                    f_out.props['_SceneChangePrev'] = 1
                    _last_sc_frame = n
                    status = "Accepted(edge_max)"
                    f_out.props['sc_reason'] = 2
                    _last_sc_status = status
                else:
                    status = "Skipped"
            # Applica min_distance
            elif above_distance_max and above_threshold:
                f_out.props['_SceneChangePrev'] = 1
                _last_sc_frame = n
                status="Accepted"
                f_out.props['sc_reason'] = 1
                _last_sc_status = status
            else:
                status="Skipped"
        else:
            status = "Rejected"

        if sc_debug:
           vsutil.HAVC_LogMessage(vsutil.MessageType.WARNING,
            f"Frame_n= {n}, luma={f_luma}, edge_diff={edge_diff}, ssim_diff={ssim_diff}, status={status}")

        return f_out

    sc_clip = clip.std.ModifyFrame(clips=[clip, sc_gray],
                                   selector=partial(set_sc_prop,
                                   ssim_diff_threshold=ssim_diff_threshold,
                                   edge_diff_threshold=edge_diff_threshold,
                                   diff_stats=diff_stats,
                                   maskdiff_stats=maskdiff_stats,
                                   sc_min_distance=sc_min_distance,
                                   sc_mult_tht=sc_mult_tht,
                                   tht_white=tht_white,
                                   tht_black=tht_black,
                                   sc_debug=sc_debug))
    """
    sc_clip = vsutil.debug_ModifyFrame(f_start=300, f_end=320, clip=clip,
                                       clips=[clip, sc_gray],
                                       selector=partial(set_sc_prop,
                                       ssim_diff_threshold=ssim_diff_threshold,
                                       edge_diff_threshold=edge_diff_threshold,
                                       diff_stats=diff_stats,
                                       maskdiff_stats=maskdiff_stats,
                                       sc_min_distance=sc_min_distance,
                                       sc_mult_tht=sc_mult_tht,
                                       tht_white=tht_white,
                                       tht_black=tht_black,
                                       sc_debug=sc_debug), silent=True)
    """
    return sc_clip

def enforce_min_scene_distance(clip: vs.VideoNode, min_distance: int = 10) -> vs.VideoNode:
    """Remove scene-change detections that are closer than min_distance frames apart.

    Iterates all frames, collects scene-change indices, applies a greedy filter keeping
    only detections separated by at least min_distance, then annotates the clip.

    :param clip:         Clip with _SceneChangePrev frame properties.
    :param min_distance: Minimum allowed gap between consecutive scene changes. Default 10.
    :return:             Clip with filtered _SceneChangePrev properties.
    """
    if min_distance <= 1:
        return clip

    # Estrai lista di frame con _SceneChangePrev
    sc_frames = []
    for i in range(len(clip)):
        f = clip.get_frame(i)
        if f.props.get('_SceneChangePrev', 0):
            sc_frames.append(i)

    # Filtra
    filtered = []
    last = -min_distance
    for n in sc_frames:
        if n - last >= min_distance:
            filtered.append(n)
            last = n

    sc_set = set(filtered)

    def apply(n, f):
        fout = f.copy()
        if n in sc_set:
            fout.props['_SceneChangePrev'] = 1
        else:
            fout.props['_SceneChangePrev'] = 0
        return fout

    return clip.std.ModifyFrame(clip, apply)

class SceneDetectionFiltered:
    """Edge-based scene detection post-filter using SSIM and histogram distance.

    Shares the same SSIM/histogram filtering logic as SceneDetection.SceneDetectFilter
    but is tuned for the edge-based detection pipeline (reads sc_reason in addition to sc_luma).
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
    _sc_prv_reason = None
    _sc_frequency = 0

    def __init__(self, sc_tht_white: float = DEF_THT_WHITE, sc_tht_black: float = DEF_THT_BLACK,
                 sc_frequency: int = 0, sc_debug: bool = False):
        """Initialise filtered scene detection state.

        :param sc_tht_white: Luma upper bound for valid scene changes. Default DEF_THT_WHITE.
        :param sc_tht_black: Luma lower bound for valid scene changes. Default DEF_THT_BLACK.
        :param sc_frequency: Minimum scene change frequency (frames). Default 0.
        :param sc_debug:     If True, log per-frame debug messages. Default False.
        """
        self._sc_debug = sc_debug
        self._sc_last_index = None
        self._sc_last_ref = None
        self._sc_prev_y = None
        self._sc_prev_luma = None
        self._sc_prv_reason = None
        self._sc_prev_index = None
        self._sc_prev_diff = 0
        self._sc_prev_hist = None
        self._sc_tht_white = sc_tht_white
        self._sc_tht_black = sc_tht_black
        self._sc_frequency = sc_frequency

        if self._sc_debug:
            vsutil.HAVC_LogMessage(vsutil.MessageType.WARNING,
                               "sc_tht_black= ", sc_tht_black,
                               ", sc_tht_white= ", sc_tht_white, ", sc_frequency= ", sc_frequency)

    def SceneDetectFilter(self, clip: vs.VideoNode, ssim_threshold: float = 0.55, min_length: int = 1) -> vs.VideoNode:
        """Post-filter scene changes using SSIM and histogram similarity (edge-based version).

        Processes the clip in batches of 5000 frames. Suppresses detections where SSIM
        indicates high similarity to the previous scene-change frame.

        :param clip:           Clip with candidate _SceneChangePrev flags and sc_luma/sc_reason props.
        :param ssim_threshold: SSIM threshold below which a detection is accepted. Default 0.55.
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

    def _scene_detect_filter_task(self, t_start: int, clip: vs.VideoNode, tht_ssim: float = 0.55, min_length: int = 1
                                  ) -> vs.VideoNode:
        """Process one batch of frames for SSIM/histogram post-filtering (edge-based version).

        :param t_start:   Absolute frame offset of this batch.
        :param clip:      Batch clip with candidate _SceneChangePrev flags.
        :param tht_ssim:  SSIM threshold; below = scene change accepted. Default 0.55.
        :param min_length: Minimum frame distance between accepted scene changes. Default 1.
        :return:          Batch clip with refined _SceneChangePrev/_SceneChangeNext properties.
        """
        def set_scenechange(n: int, f: vs.VideoFrame, t_start: int, clip: vs.VideoNode, ssim_tht: float,
                            tht_white: float, tht_black, min_length: int = 1) -> vs.VideoFrame:
            fout = f.copy()
            f_luma: float = fout.props['sc_luma']
            f_reason: int = fout.props['sc_reason']

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
                self._sc_prev_reason = 0

            is_scenechange = fout.props['_SceneChangePrev'] == 1 or t_n == 0

            if is_scenechange and self._sc_last_index is None:
                fout.props['_SceneChangePrev'] = 1
                fout.props['_SceneChangeNext'] = 0
                self._sc_last_index = t_n
                self._sc_prev_y = y_last
                self._sc_prev_luma = f_luma
                self._sc_prev_reason = f_reason
                self._sc_prev_hist = self._calc_histogram(y_img)
                if self._sc_debug:
                    vsutil.HAVC_LogMessage(vsutil.MessageType.WARNING,
                                           "SC=[New], Frame_n= ", t_n, ", PrvFrame= ", self._sc_last_index,
                                           ", SSIM= ", -1, ", Hist= ", -1, ", Luma= ", f_luma, ", ScReason= 1")
                return fout

            if not is_scenechange:
                return fout

            sc_reason = 0

            if is_scenechange and n > 0 and (t_n - self._sc_last_index) < min_length:
                if min_length > 1 and n > 1 and self._sc_prev_luma >= DEF_THT_BLACK_MIN > f_luma:
                    if self._sc_debug:
                        vsutil.HAVC_LogMessage(vsutil.MessageType.WARNING,
                                               "SC=[Skip], Frame_n= ", t_n, ", PrvFrame= ", self._sc_last_index,
                                               ", SSIM= ", -1, ", Hist = ", -1, ", Luma= ", f_luma, ", ScReason= -1")
                    fout.props['_SceneChangePrev'] = 0
                    fout.props['_SceneChangeNext'] = 0
                    return fout
                else:
                    sc_reason = 4

            y_hist = self._calc_histogram(y_img)

            if ssim_tht == 1:
                ssim_score = 1
                hist_score = 1
                scene_change = tht_black < f_luma < tht_white
                sc_reason = (sc_reason + 1) if scene_change else 0
            elif n < clip.num_frames:
                ssim_score = round(structural_similarity(y_last, self._sc_prev_y, full=False), 4)
                hist_compare = cv2.compareHist(H1=self._sc_prev_hist, H2=y_hist, method=cv2.HISTCMP_HELLINGER)
                hist_score = round(1 - hist_compare, 4)
                if f_reason > 1 and self._sc_prev_reason < 2:
                    # override on reason and luma
                    scene_change = tht_black < f_luma < tht_white
                elif ssim_score < ssim_tht and hist_score < DEF_HIST_SCORE_HIGH:
                    scene_change = tht_black < f_luma < tht_white
                    # override on ratio and luma
                    if scene_change and sc_reason == 0 and self._sc_frequency > 1:
                        scene_change = (scene_change and not (f_luma < DEF_THT_BLACK_FREQ))
                    sc_reason = (sc_reason + 1) if scene_change else 0
                elif ssim_score >= DEF_SSIM_SCORE_EQUAL and self._sc_prev_luma < DEF_THT_BLACK_MIN <= f_luma:
                    # force scene change to get better frame
                    scene_change = tht_black < f_luma < tht_white
                    sc_reason = (sc_reason + 2) if scene_change else 0
                elif ssim_score >= DEF_SSIM_SCORE_EQUAL and hist_score < DEF_HIST_SCORE_EQUAL:
                    scene_change = DEF_THT_BLACK_MIN < f_luma < DEF_THT_WHITE_MIN
                    sc_reason = (sc_reason + 3) if scene_change else 0
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
                                           ", SSIM= ", ssim_score, ", Hist= ", hist_score, ", Luma= ", f_luma,
                                           ", ScReason= ", sc_reason)
                fout.props['_SceneChangePrev'] = 1
                fout.props['_SceneChangeNext'] = 0
                self._sc_last_index = t_n
                self._sc_prev_y = y_last
                self._sc_prev_hist = y_hist
                self._sc_prev_luma = f_luma
            else:
                if self._sc_debug:
                    vsutil.HAVC_LogMessage(vsutil.MessageType.WARNING,
                                           "SC=[Skip], Frame_n: ", t_n, ", PrvFrame= ", self._sc_last_index,
                                           ", SSIM= ", ssim_score, ", Hist = ", hist_score, ", Luma= ", f_luma,
                                           ", ScReason= ", sc_reason)
                fout.props['_SceneChangePrev'] = 0
                fout.props['_SceneChangeNext'] = 0

            return fout

        """
        sc = vsutil.debug_ModifyFrame(0, 250, clip, clips=[clip],
                                      selector=partial(set_scenechange, t_start=t_start, clip=clip, ssim_tht=tht_ssim,
                                                       tht_white=self._sc_tht_white, tht_black=self._sc_tht_black,
                                                       min_length=min_length))
        """ 

        sc = clip.std.ModifyFrame(clips=[clip],
                                  selector=partial(set_scenechange, t_start=t_start, clip=clip, ssim_tht=tht_ssim,
                                                   tht_white=self._sc_tht_white, tht_black=self._sc_tht_black,
                                                   min_length=min_length))

        return sc

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