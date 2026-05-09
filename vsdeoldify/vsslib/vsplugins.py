"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-10-09
version: 
LastEditors: Dan64
LastEditTime: 2024-10-26
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Utility functions to load Vapoursynth plugins dynamically.
"""

import vapoursynth as vs
from pathlib import Path
from vsdeoldify.vsslib.mcomb import vs_combine_models
from vsdeoldify.vsslib.vsfilters import vs_tweak
from vsdeoldify.vsslib.vsresize import resize_min_HW
from vsdeoldify.vsslib.vsutils import HAVC_LogMessage, MessageType, frame_to_image

from vsdeoldify.vsslib.constants import *

from vsdeoldify.vsslib.__int__ import *

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Utility functions to load Vapoursynth plugins dynamically.
"""
def load_TCanny_plugin() -> bool:
    """
    Ensures TCanny VapourSynth plugin is loaded.
    URL: https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny
    """

    plugin_path = os.path.normpath(os.path.join(support_dir, "TCanny.dll"))

    try:
        if hasattr(vs.core, 'tcanny') and hasattr(vs.core.akarin, 'TCanny'):
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION,f"[INFO] Plugin 'TCanny' already loaded.")
            return True
        else:
            vs.core.std.LoadPlugin(path=plugin_path)
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION, f"[INFO] Plugin 'TCanny' loaded from: {plugin_path}")
            return True
    except Exception as error:
        HAVC_LogMessage(MessageType.WARNING,"[WARNING] Plugin 'TCanny': check/load failed ->", str(error))
        return False

def load_Akarin_plugin() -> bool:
    """
    Ensures Akarin VapourSynth plugin is loaded.
    URL: https://github.com/AkarinVS/vapoursynth-plugin
    """

    plugin_path = os.path.normpath(os.path.join(support_dir, "akarin.dll"))

    try:
        if hasattr(vs.core, 'akarin') and hasattr(vs.core.akarin, 'Expr'):
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION,f"[INFO] Plugin 'Akarin' already loaded.")
            return True
        else:
            vs.core.std.LoadPlugin(path=plugin_path)
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION, f"[INFO] Plugin 'Akarin' loaded from: {plugin_path}")
            return True
    except Exception as error:
        HAVC_LogMessage(MessageType.WARNING,"[WARNING] Plugin 'Akarin': check/load failed ->", str(error))
        return False

def load_SCXvid_plugin() -> bool:
    """
    Ensures SCXvid VapourSynth plugin is loaded.
    URL: https://github.com/dubhater/vapoursynth-scxvid
    """

    plugin_path = os.path.normpath(os.path.join(support_dir, "libscxvid.dll"))

    try:
        if hasattr(vs.core, 'scxvid') and hasattr(vs.core.scxvid, 'Scxvid'):
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION,f"[INFO] Plugin 'SCXvid' already loaded.")
            return True
        else:
            vs.core.std.LoadPlugin(path=plugin_path)
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION, f"[INFO] Plugin 'SCXvid' loaded from: {plugin_path}")
            return True
    except Exception as error:
        HAVC_LogMessage(MessageType.WARNING,"[WARNING] Plugin 'SCXvid': check/load failed ->", str(error))
        return False

def load_MVTool_plugin() -> bool:
    """
    Ensures MVTool VapourSynth plugin is loaded.
    URL: https://github.com/dubhater/vapoursynth-mvtools
    """

    plugin_path = os.path.normpath(os.path.join(support_dir, "libmvtools.dll"))

    try:
        if hasattr(vs.core, 'mv') and hasattr(vs.core.mv, 'SCDetection'):
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION,f"[INFO] Plugin 'MVTool' already loaded.")
            return True
        else:
            vs.core.std.LoadPlugin(path=plugin_path)
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION, f"[INFO] Plugin 'MVTool' loaded from: {plugin_path}")
            return True
    except Exception as error:
        HAVC_LogMessage(MessageType.WARNING,"[WARNING] Plugin 'MVTool': check/load failed ->", str(error))
        return False

def load_Zsmooth_plugin() -> bool:
    """
    Ensures Zsmooth VapourSynth plugin is loaded.
    URL: https://github.com/adworacz/zsmooth?tab=readme-ov-file#median
    """

    plugin_path = os.path.normpath(os.path.join(Zsmooth_dir, "zsmooth.dll"))

    try:
        if hasattr(vs.core, 'zsmooth') and hasattr(vs.core.zsmooth, 'TemporalMedian'):
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION,f"[INFO] Plugin 'Zsmooth' already loaded.")
            return True
        else:
            vs.core.std.LoadPlugin(path=plugin_path)
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION, f"[INFO] Plugin 'Zsmooth' loaded from: {plugin_path}")
            return True
    except Exception as error:
        HAVC_LogMessage(MessageType.WARNING,"[WARNING] Plugin 'Zsmooth': check/load failed ->", str(error))
        return False

def load_Retinex_plugin() -> bool:
    """
    Ensures Retinex VapourSynth plugin is loaded.
    URL: https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Retinex
    """

    plugin_path = os.path.normpath(os.path.join(Retinex_dir, "Retinex.dll"))

    try:
        if hasattr(vs.core, 'retinex') and hasattr(vs.core.retinex, 'MSRCP'):
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION,f"[INFO] Plugin 'Retinex' already loaded.")
            return True
        else:
            vs.core.std.LoadPlugin(path=plugin_path)
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION, f"[INFO] Plugin 'Retinex' loaded from: {plugin_path}")
            return True
    except Exception as error:
        HAVC_LogMessage(MessageType.WARNING,"[WARNING] Plugin 'Retinex': check/load failed ->", str(error))
        return False

def load_SCDetect_plugin() -> bool:
    """
    Ensures SCDetect VapourSynth plugin is loaded.
    URL: https://github.com/vapoursynth/vs-miscfilters-obsolete
    """

    plugin_path = os.path.normpath(os.path.join(MiscFilter_dir, "MiscFilters.dll"))

    try:
        if hasattr(vs.core, 'misc') and hasattr(vs.core.misc, 'SCDetect'):
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION,"[INFO] Plugin 'SCDetect' already loaded.")
            return True
        else:
            vs.core.std.LoadPlugin(path=plugin_path)
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION, f"[INFO] Plugin 'SCDetect' loaded from: {plugin_path}")
            return True
    except Exception as error:
        HAVC_LogMessage(MessageType.WARNING, "[WARNING] Plugin 'SCDetect': check/load failed ->", str(error))
        return False

def load_ReduceFlicker_plugin() -> bool:
    """
    Ensures ReduceFlicker VapourSynth plugin is loaded.
    URL: https://github.com/AmusementClub/ReduceFlicker
    """

    plugin_path = os.path.normpath(os.path.join(ReduceFlicker_dir, "ReduceFlicker.dll"))

    try:
        if hasattr(vs.core, 'rdfl') and hasattr(vs.core.rdfl, 'ReduceFlicker'):
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION,"[INFO] Plugin 'ReduceFlicker' already loaded.")
            return True
        else:
            vs.core.std.LoadPlugin(path=plugin_path)
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION, f"[INFO] Plugin 'ReduceFlicker' loaded from: {plugin_path}")
            return True
    except Exception as error:
        HAVC_LogMessage(MessageType.WARNING,"[WARNING] Plugin 'ReduceFlicker': check/load failed ->", str(error))
        return False

def load_LSMASHSource_plugin() -> bool:
    """
    Ensures LSMASHSource VapourSynth plugin is loaded.
    URL: https://github.com/AkarinVS/L-SMASH-Works
    """

    plugin_path = os.path.normpath(os.path.join(LSMASHSource_dir, "LSMASHSource.dll"))

    try:
        if hasattr(vs.core, 'lsmas') and hasattr(vs.core.lsmas, 'LWLibavSource'):
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION,"[INFO] Plugin 'LSMASHSource' already loaded.")
            return True
        else:
            vs.core.std.LoadPlugin(path=plugin_path)
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION, f"[INFO] Plugin 'LSMASHSource' loaded from: {plugin_path}")
            return True
    except Exception as error:
        HAVC_LogMessage(MessageType.WARNING,"[WARNING] Plugin 'LSMASHSource': check/load failed ->", str(error))
        return False


def load_TimeCube_plugin() -> bool:
    """
    Ensures TimeCube VapourSynth plugin is loaded.
    URL: https://github.com/sekrit-twc/timecube
    """

    plugin_path = os.path.normpath(os.path.join(TimeCube_dir, "vscube.dll"))

    try:
        if hasattr(vs.core, 'timecube') and hasattr(vs.core.timecube, 'Cube'):
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION,"[INFO] Plugin 'TimeCube' already loaded.")
            return True
        else:
            vs.core.std.LoadPlugin(path=plugin_path)
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                HAVC_LogMessage(MessageType.INFORMATION, f"[INFO] Plugin 'TimeCube' loaded from: {plugin_path}")
            return True
    except Exception as error:
        HAVC_LogMessage(MessageType.WARNING,"[WARNING] Plugin 'TimeCube': check/load failed ->", str(error))
        return False

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Wrapper to ReduceFlicker
"""

def vs_reduce_flicker(clip: vs.VideoNode, strength: int = 2, aggressive: int = 0) -> vs.VideoNode:
    """Reduce temporal brightness flickering using the ReduceFlicker VapourSynth plugin.

    :param clip:       Input clip, any format supported by ReduceFlicker.dll.
    :param strength:   Strength of flicker reduction [1, 3]. Default 2.
    :param aggressive: Enable aggressive mode (0 = off, 1 = on). Default 0.
    :return:           De-flickered clip.
    """
    load_ReduceFlicker_plugin()

    try:
        clip = vs.core.rdfl.ReduceFlicker(clip=clip, strength=strength, aggressive=aggressive)
    except Exception as error:
        raise vs.Error("vs_retinex: plugin 'ReduceFlicker.dll' not properly loaded/installed -> " + str(error))

    return clip

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Wrapper to TimeCube
"""

def vs_timecube_load(clip: vs.VideoNode, lut_effect: int = DEF_LUT_Exploration) -> vs.VideoNode:
    """Load and apply a .cube LUT file to a clip via the TimeCube VapourSynth plugin.

    Selects one of 12 built-in LUT presets (0–11) from the TimeCube colour directory.
    Returns the original clip unchanged if the .cube file is not found.

    :param clip:       RGB24 input clip.
    :param lut_effect: LUT preset ID (0–11), mapped to constants DEF_LUT_*. Default: DEF_LUT_Exploration.
    :return:           Clip with LUT applied.
    """
    load_TimeCube_plugin()

    f_name: str = ""
    match lut_effect:
        case 0:
            f_name = os.path.normpath(os.path.join(TimeCube_dir, "color", "Stockpresets - Forest Film.cube"))
        case 1:
            f_name = os.path.normpath(os.path.join(TimeCube_dir, "color", "Presetpro - City Skyline.cube"))
        case 2:
            f_name = os.path.normpath(os.path.join(TimeCube_dir, "color", "Presetpro - Exploration.cube"))
        case 3:
            f_name = os.path.normpath(os.path.join(TimeCube_dir, "color", "Presetpro - FUJ Film.cube"))
        case 4:
            f_name = os.path.normpath(os.path.join(TimeCube_dir, "color", "Presetpro - Hollywood.cube"))
        case 5:
            f_name = os.path.normpath(os.path.join(TimeCube_dir, "color", "Presetpro - Classic Film.cube"))
        case 6:
            f_name = os.path.normpath(os.path.join(TimeCube_dir, "color", "Presetpro - Warm Haze.cube"))
        case 7:
            f_name = os.path.normpath(os.path.join(TimeCube_dir, "color", "Presetpro - HDR Color.cube"))
        case 8:
            f_name = os.path.normpath(os.path.join(TimeCube_dir, "color", "Presetpro - Amber Light.cube"))
        case 9:
            f_name = os.path.normpath(os.path.join(TimeCube_dir, "color", "Presetpro - Blue Mist.cube"))
        case 10:
            f_name = os.path.normpath(os.path.join(TimeCube_dir, "color", "Presetpro - Vintage Fox.cube"))
        case 11:
            f_name = os.path.normpath(os.path.join(TimeCube_dir, "color", "Presetpro - Flat Pop.cube"))

    if not Path(f_name).is_file():
        HAVC_LogMessage(MessageType.INFORMATION, f"LUT cube file: {f_name} not found!")
        return clip

    try:
        clip_new = vs.core.timecube.Cube(clip=clip, cube=f_name)
    except Exception as error:
        raise vs.Error("vs_timecube: plugin 'vscube.dll' not properly loaded/installed -> " + str(error))

    return clip_new

def vs_timecube(clip: vs.VideoNode, strength: float = 1.0, lut_effect: int = DEF_LUT_Exploration, factors: list = None) -> vs.VideoNode:
    """Apply a TimeCube LUT with optional per-LUT colour tweaks and blend strength.

    Loads the LUT via vs_timecube_load, applies preset hue/sat/bright/cont/gamma
    adjustments tuned for each LUT, then blends the result back with the original
    clip at the requested strength.  If factors is provided it overrides the preset
    adjustments.

    :param clip:       RGB24 input clip.
    :param strength:   Blend weight of the LUT result [0, 1]. 0 = bypass, 1 = full LUT.
    :param lut_effect: LUT preset ID (0–11). Default: DEF_LUT_Exploration.
    :param factors:    Optional override list [hue, sat, bright, cont, gamma].
    :return:           Colour-graded clip.
    """
    if strength == 0:
        return clip   # nothing to do

    clip_new = vs_timecube_load(clip, lut_effect)

    hue: float = 0; sat: float = 1; bright: float = 0; cont: float = 1; gamma: float = 1

    if factors is None:
        match lut_effect:
            case 0:
                sat = 0.70; hue = 10
            case 1:
                cont = 0.90; sat = 0.65; hue = -3; bright = 1; gamma = 1.05
            case 2:
                sat = 1.05; cont = 1.05; gamma = 0.95; hue = 10; bright = -1
            case 3:
                sat = 0.80; hue = 10
            case 4:
                sat = 0.75; hue = 10
            case 5:
                sat = 0.80
            case 6:
                sat = 0.75
            case 7:
                sat = 0.95
            case 8:
                sat = 0.40; hue = 10; bright = 5
            case 9:
                sat = 0.80; hue = 3; bright = -1
            case 10:
                sat = 0.80; hue = 3; bright = 1
            case 11:
                sat = 0.80; hue = -2; bright = 0
    else:
        hue = factors[0]
        sat = factors[1]
        bright = factors[2]
        cont = factors[3]
        gamma = factors[4]

    clip_new = vs_tweak(clip_new, cont = cont, sat = sat, hue = hue, bright = bright, gamma = gamma)

    if strength == 1:
        return clip_new

    if lut_effect == 8:
        clip_new = vs_combine_models(clip_a=clip, clip_b=clip_new, method=7, clipb_weight=strength,
                                           CMC_p=[0.15, True, 25, 25])
    else:
        clip_new = vs.core.std.Merge(clipa=clip, clipb=clip_new, weight=strength)

    return clip_new



