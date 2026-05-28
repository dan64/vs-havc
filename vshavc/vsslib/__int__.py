"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2025-10-04
version:
LastEditors: Dan64
LastEditTime: 2026-02-01
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
HAVC functions library.
"""
import os

# Hybrid paths for plugins used by HAVC
vsslib_dir: str = os.path.dirname(os.path.realpath(__file__))

support_dir: str = os.path.join(vsslib_dir, "..", "..", "..", "..", "..", "vsfilters", "Support")

# Path for Zsmooth.Median:
Zsmooth_dir: str = os.path.join(vsslib_dir, "..", "..", "..", "..", "..", "vsfilters", "DenoiseFilter", "ZSmooth")

# Path for ReduceFlicker:
ReduceFlicker_dir: str = os.path.join(vsslib_dir, "..", "..", "..", "..", "..", "vsfilters", "FrameFilter", "ReduceFlicker")

# Path for Retinex:
Retinex_dir: str = os.path.join(vsslib_dir, "..", "..", "..", "..", "..", "vsfilters", "ColorFilter", "Retinex")

# Path for SCDetect:
MiscFilter_dir: str = os.path.join(vsslib_dir, "..", "..", "..", "..", "..", "vsfilters", "MiscFilter", "MiscFilters")

# Path for LSMASHSource:
LSMASHSource_dir: str = os.path.join(vsslib_dir, "..", "..", "..", "..", "..", "vsfilters", "SourceFilter", "LSmashSource")

# Path for TimeCube:
TimeCube_dir: str = os.path.join(vsslib_dir, "..", "..", "..", "..", "..", "vsfilters", "ColorFilter", "TimeCube")
