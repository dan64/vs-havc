"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2024-11-20
version:
LastEditors: Dan64
LastEditTime: 2026-04-24
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Constants for vs-havc functions.
"""
DEF_CONNECT_TIMEOUT: int = 5
DEF_CALL_TIMEOUT: int = 180
DEF_LEVEL_NONE: int = 0
DEF_LEVEL_INFO: int = 1
DEF_LEVEL_DEBUG: int = 2
DEF_DEBUG_LEVEL: int = 0
DEF_MAX_RESIZE: int = 512
DEF_HAVC_METHOD_PLACEBO: int = 10
DEF_CMC_p: list = [0.15, True, 20, 24]
DEF_LMM_p: list = [0.15, 0.65, 1.0]
DEF_ALM_p: list = [0.8, 1.0, 0.15]
DEF_CRT_p: list = [0.8, 30, 2, False, 0, 0]
DEF_TWEAK_p: list = [0.0, 1.0, 2.5, True, 0.3, 0.6, 1.5, 0.5]
DEF_THT_WHITE: float = 0.70
DEF_THT_BLACK: float = 0.10
DEF_RETINEX_DARK = 0.20
DEF_RETINEX_BRIGHT = 0.80
DEF_STANDARD_DARK = 0.22
DEF_STANDARD_BRIGHT = 0.78
DEF_LUT_Forest_Film: int = 0
DEF_LUT_City_Skyline: int = 1
DEF_LUT_Exploration: int = 2
DEF_LUT_FUJ_Film: int = 3
DEF_LUT_Hollywood: int = 4
DEF_LUT_Classic_Film: int = 5
DEF_LUT_Warm_Haze: int = 6
DEF_LUT_HDR_Color: int = 7
DEF_LUT_Amber_Light: int = 8
DEF_LUT_Blue_Mist: int = 9
DEF_LUT_Vintage_Fox: int = 10
DEF_LUT_Flat_Pop: int = 11
DEF_THT_BLACK_FREQ: float = 0.14
DEF_THT_BLACK_MIN: float = 0.19
DEF_THT_WHITE_MIN: float = 0.70
DEF_THT_DARK_BLACK: float = 0.15
DEF_THT_BRIGHT_WHITE: float = 0.70
DEF_ADAPTIVE_RATIO_LO: float = 1.02
DEF_ADAPTIVE_RATIO_MED: float = 1.12
DEF_ADAPTIVE_RATIO_HI: float = 1.20
DEF_ADAPTIVE_RATIO_RF: float = 2.0
DEF_ADAPTIVE_RATIO_VHI: float = 15.0
DEF_SSIM_SCORE_EQUAL: float = 0.69
DEF_HIST_SCORE_EQUAL: float = 0.70
DEF_HIST_SCORE_HIGH: float = 0.95
DEF_MERGE_LOW_WEIGHT: float = 0.20
DEF_STABLE_WEIGHT: float = 0.50
DEF_ARTISTIC_WEIGHT: float = 0.50
DEF_EXPORT_FORMAT: str = 'jpg'
DEF_JPG_QUALITY: int = 95
DEF_THRESHOLD: float = 0.10
DEF_MIN_CPUID_RESIZE: int = 3        # min cpu_id to force chroma_resize
DEF_MIN_FREQ: int = 10               # MIN reference frames frequency for deep-remaster
DEF_MAX_FREQ: int = 15               # MAX reference frames frequency for deep-remaster
DEF_SC_MIN_DISTANCE: int = 15
DEF_MAX_MEMORY_FRAMES: int = 1000    # theoretically MAX_MEMORY must be < 95000, safe = 1000
DEF_MAX_RF_FRAMES: int = 200         # MAX number of reference frames to use for deep-remster
DEF_NUM_RF_FRAMES: int = 10          # default number of reference frames to use for deep-remster
DEF_MIN_RF_FRAMES: int = 4           # MIN number of reference frames to use for deep-remster
DEF_MAX_BUFFER_SIZE: int = 500       # number of frames to scan for searching RF for deep-remster
DEF_MAX_XREF_BUFFER: int = 500       # number of frames to scan for searching RF for colormnet
DEF_MAX_XRF_FRAMES: int = 250        # MAX number of reference frames to load at start for colormnet
DEF_MAX_XREF_WINDOW: int = 20        # number of forward reference frames for colormnet
DEF_NUM_XRF_FRAMES: int = 20         # default number of reference frames to load at start for colormnet
DEF_MIN_XRF_FRAMES: int = 2          # MIN number of reference frames to load at start for colormnet
DEF_FUTURE_FRAME_WEIGHT: float = 0.5
DEF_XRF_WINDOW_SIZE: int = 20        # default window_size for cmnet2 (replaces max_memory_frames semantics)
DEF_XRF_MAX_WINDOW_SIZE: int = 40    # max window_size for initial loading in cmnet2
DEF_XRF_HALF_WINDOW_SIZE: int = 20
DEF_XRF_SLIDE_PERCENT: float = 0.2   # SLIDE_STEP = max(1, round(window_size * DEF_XRF_SLIDE_PERCENT + 0.5))
DEF_BATCH_SIZE: int = 2
DEF_VIVID_HUE_LOW: float = 3.0
DEF_VIVID_SAT_HIGH: float = 1.30
DEF_VIVID_HUE_HIGH: float = 5.0
DEF_VIVID_SAT_LOW: float = 1.15
DEF_MIN_COLOR_ALPHA: float = 1.0
DEF_MAX_COLOR_ALPHA: float = 10.0
DEF_RECOVER_WEIGHT: float = 0.99





