"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2025-11-01
version:
LastEditors: Dan64
LastEditTime: 2026-05-01
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Utility functions to:

Slice the original clip into 4 overlapping tiles (2×2 grid), each slightly larger than a quarter,
so adjacent tiles share a border.

Reconstruct by blending the overlapping regions smoothly using linear feathering.

It will be used the akarin plugin to generate position-based blend masks for seamless merging.

"""
from dataclasses import dataclass
from typing import List, Optional
import vapoursynth as vs
from vshavc.vsslib import vsresize
from vshavc.vsslib.vsplugins import load_Akarin_plugin
from vshavc.vsslib.vsutils import frame_to_image

@dataclass
class ClipTiles:
    """ClipTiles dataclass.
       Parameters:
         clip_orig: original clip
         tiles: [tl, tr, bl, br] — each larger than H/2 x W/2 due to overlap
         base_tile_w: base tile width (without extra padding)
         base_tile_h: base tile height (without extra padding)
         overlap_x: actual horizontal overlap pixels
         overlap_y: actual vertical overlap pixels
    """
    clip_orig: Optional[vs.VideoNode]  # original clip
    tiles: List[vs.VideoNode]          # [tl, tr, bl, br] — each larger than H/2 x W/2 due to overlap
    base_tile_w: int                   # base tile size (without extra padding)
    base_tile_h: int
    overlap_x: int                     # actual overlap in pixels
    overlap_y: int

core = vs.core

def vs_slice_into_2x2_overlapping_tiles(clip: vs.VideoNode, overlap_x: int = 32, overlap_y: int = 32) -> ClipTiles:
    """
    Slices the input clip into 4 overlapping tiles (2x2 grid).

    Assumptions:
        - Original frame size: H × W (can be odd or even).
        - Each tile will be larger than (H//2) × (W//2) to include overlap.
        - Overlap = horizontal/vertical overlap pixels.
        - We’ll pad the original clip if needed to ensure clean tiling.

    Args:
        clip: clip to be sliced in 4 tiles
        overlap_x: horizontal overlap pixels size
        overlap_y: vertical overlap pixels size

    Returns:
        class ClipTiles with items:
            - clip_orig: original clip
            - tiles: [tl, tr, bl, br] — each larger than H/2 x W/2 due to overlap
            - base_tile_w: base tile width (without extra padding)
            - base_tile_h: base tile height (without extra padding)
            - overlap_x: actual horizontal overlap pixels
            - overlap_y: actual vertical overlap pixels
    """
    w, h = clip.width, clip.height

    # Base tile size (ideal quarter)
    base_tile_w = (w + 1) // 2  # ceil(w/2) to cover all pixels
    base_tile_h = (h + 1) // 2

    # Pad the original clip to allow full tile extraction at edges
    # Padding ensures we can crop full-sized tiles even at borders
    overlap_x = (overlap_x // 2) * 2  # Round to the nearest even number (downward)
    overlap_y = (overlap_y // 2) * 2  # Round to the nearest even number (downward)
    pad_right = overlap_x
    pad_bottom = overlap_y

    padded = core.std.AddBorders(clip, left=0, right=pad_right, top=0, bottom=pad_bottom)

    # Define crop positions for 4 tiles
    # Top-left
    tl = core.std.CropAbs(padded, left=0, top=0, width=base_tile_w + overlap_x, height=base_tile_h + overlap_y)
    # Top-right
    tr = core.std.CropAbs(padded, left=base_tile_w - overlap_x, top=0, width=base_tile_w + overlap_x,
                          height=base_tile_h + overlap_y)
    # Bottom-left
    bl = core.std.CropAbs(padded, left=0, top=base_tile_h - overlap_y, width=base_tile_w + overlap_x,
                          height=base_tile_h + overlap_y)
    # Bottom-right
    br = core.std.CropAbs(padded, left=base_tile_w - overlap_x, top=base_tile_h - overlap_y,
                          width=base_tile_w + overlap_x, height=base_tile_h + overlap_y)

    clip_tiles = ClipTiles(clip_orig=clip,
                           tiles=[tl, tr, bl, br],
                           base_tile_w=base_tile_w,
                           base_tile_h=base_tile_h,
                           overlap_x=overlap_x,
                           overlap_y=overlap_y)

    return clip_tiles

def vs_slice_into_2_horizontal_tiles(clip: vs.VideoNode, overlap_x: int = 32) -> ClipTiles:
    """
    Slices the input clip into 2 horizontal overlapping tiles.

    Assumptions:
        - Original frame size: H × W (can be odd or even).
        - Each tile will be larger than H × (W//2) to include overlap.
        - Overlap = horizontal overlap pixels
        - We’ll pad the original clip if needed to ensure clean tiling.

    Args:
        clip: clip to be sliced in 4 tiles
        overlap_x: horizontal overlap pixels size

    Returns:
        class ClipTiles with items:
            - clip_orig: original clip
            - tiles: [tl, tr] — each larger than H/2 x W/2 due to overlap
            - base_tile_w: base tile width (without extra padding)
            - base_tile_h: original height
            - overlap_x: actual horizontal overlap pixels
            - overlap_y: 0
    """
    w, h = clip.width, clip.height

    # Base tile size (ideal quarter)
    base_tile_w = (w + 1) // 2  # ceil(w/2) to cover all pixels

    # Pad the original clip to allow full tile extraction at edges
    # Padding ensures we can crop full-sized tiles even at borders
    overlap_x = (overlap_x // 2) * 2  # Round to the nearest even number (downward)
    pad_right = overlap_x
    pad_bottom = 0

    padded = core.std.AddBorders(clip, left=0, right=pad_right, top=0, bottom=pad_bottom)

    # Define crop positions for the 2 horizontal tiles
    # Top-left
    tl = core.std.CropAbs(padded, left=0, top=0, width=base_tile_w + overlap_x, height= h)
    # Top-right:
    tr = core.std.CropAbs(padded, left=base_tile_w - overlap_x, top=0, width=base_tile_w + overlap_x, height= h)

    clip_tiles = ClipTiles(clip_orig=clip,
                           tiles=[tl, tr],
                           base_tile_w=base_tile_w,
                           base_tile_h= h,
                           overlap_x=overlap_x,
                           overlap_y=0)

    return clip_tiles


def vs_reconstruct_from_2x2_overlapping_tiles(clip_tiles: ClipTiles,
                                              blend_weight: float = 0.5,  # 0.5 = 50% from right/bottom
                                              chroma_resize: bool = False) -> vs.VideoNode:
    """
    Reconstructs the original clip from 4 overlapping filtered tiles,
    using smooth linear blending in overlap zones.

    Slicing:
        - Each tile = (base_h + overlap_y) × (base_w + overlap_x)
        - Tiles are extracted from a padded version of the original to ensure full coverage.
        - Overlap ensures neighboring tiles share context.

    Blending:
        - Horizontal blend: In the central vertical strip of width 2×overlap_x, we fade from left to right.
        - Vertical blend: In the central horizontal strip of height 2×overlap_y, we fade from top to bottom.
        - Uses linear ramp via akarin.Expr() for smooth transitions.
        - Final output is cropped to original dimensions

    Args:
        clip_tiles: ClipTiles with items
            - tiles: list of sliced clips
            - original_width: original width
            - original_height: original height
            - base_tile_w: base tile width (without extra padding)
            - base_tile_h: base tile height (without extra padding)
            - overlap_x: actual horizontal overlap pixels
            - overlap_y: actual vertical overlap pixels
        blend_weight: fixed weight to merge the overlapping region, if == 0 is performed a merge with linear
                      increasing weight from 0 to 1 in the overlapping region for smooth transitions, default = 0.5
        chroma_resize: if True will be performed a chroma Resize. The Y plane of reconstructed clip will be 
                       replaced by the Y plane of the original clip.
    Return:
       clip after reconstruction = original size H × W.
    """

    load_Akarin_plugin()

    tiles = clip_tiles.tiles
    original_width = clip_tiles.clip_orig.width
    original_height = clip_tiles.clip_orig.height
    base_tile_w = clip_tiles.base_tile_w
    base_tile_h = clip_tiles.base_tile_h
    overlap_x = clip_tiles.overlap_x
    overlap_y = clip_tiles.overlap_y

    tl, tr, bl, br = tiles

    # Blend rows
    top_half = _blend_horizontal(tl, tr, overlap_x, base_tile_w, blend_weight)

    bottom_half = _blend_horizontal(bl, br, overlap_x, base_tile_w, blend_weight)

    # Blend columns
    full = _blend_vertical(top_half, bottom_half, overlap_y, base_tile_h, blend_weight)

    # Crop to original size
    if full.width != original_width or full.height != original_height:
        full = core.std.CropAbs(full, width=original_width, height=original_height)

    if chroma_resize:
        full = vsresize.resize_to_chroma(clip_tiles.clip_orig, full)

    return full

def vs_reconstruct_from_2_horizontal_tiles(clip_tiles: ClipTiles,
                                            blend_weight: float = 0.5,  # 0.5 = 50% from right/bottom
                                            chroma_resize: bool = False) -> vs.VideoNode:
    """
    Reconstructs the original clip from 2 overlapping horizontal filtered tiles,
    using smooth linear blending in overlap zones.

    Slicing:
        - Each tile = base_h × (base_w + overlap_x)
        - Tiles are extracted from a padded version of the original to ensure full coverage.
        - Overlap ensures neighboring tiles share context.

    Blending:
        - Horizontal blend: In the central vertical strip of width 2×overlap_x, we fade from left to right.
        - Uses linear ramp via akarin.Expr() for smooth transitions.
        - Final output is cropped to original dimensions

    Args:
        clip_tiles: ClipTiles with items
            - tiles: list of sliced clips
            - base_tile_w: base tile width (without extra padding)
            - base_tile_h: base tile height
            - overlap_x: actual horizontal overlap pixels
            - overlap_y: 0
        blend_weight: fixed weight to merge the overlapping region, if == 0 is performed a merge with linear
                      increasing weight from 0 to 1 in the overlapping region for smooth transitions, default = 0.5
        chroma_resize: if True will be performed a chroma Resize. The Y plane of reconstructed clip will be 
                       replaced by the Y plane of the original clip.

    Return:
       clip after reconstruction = original size H × W.
    """

    load_Akarin_plugin()

    tiles = clip_tiles.tiles
    original_width = clip_tiles.clip_orig.width
    original_height = clip_tiles.clip_orig.height
    base_tile_w = clip_tiles.base_tile_w
    overlap_x = clip_tiles.overlap_x

    tl, tr = tiles

    # Blend rows
    full = _blend_horizontal(tl, tr, overlap_x, base_tile_w, blend_weight)

    # Crop to original size
    if full.width != original_width or full.height != original_height:
        full = core.std.CropAbs(full, width=original_width, height=original_height)

    if chroma_resize:
        full = vsresize.resize_to_chroma(clip_tiles.clip_orig, full)

    return full


def _make_horizontal_blend_mask_akarin(width: int, height: int, overlap: int, base_w: int, weight: float) -> vs.VideoNode:
    """Build a horizontal blend mask using akarin.Expr for tile seam blending.

    For weight==0, generates a linear ramp from 0 to 255 over the 2×overlap transition zone.
    For weight>0, generates a constant value (weight*255) in the transition zone.

    :param width:   Mask width in pixels (= 2 × base_w).
    :param height:  Mask height in pixels.
    :param overlap: Half-width of the blending transition zone.
    :param base_w:  Base tile width (centre of transition at x == base_w).
    :param weight:  Blend weight in the transition zone [0, 1]. 0 = linear ramp.
    :return:        GRAY8 mask clip.
    """
    mask = core.std.BlankClip(width=width, height=height, format=vs.GRAY8)

    mask_val = int(round(weight * 255))
    start = base_w - overlap
    start1 = start + 1  # x < start1  <=> x <= start
    end = base_w + overlap
    end1 = end - 1  # x > end1    <=> x >= end

    if mask_val == 0: # linear merge
        # Expression: if x < start1 → 0, else if x > end1 → 255, else → (x - start) * 255 / overlap
        expr = f"X {start1} < 0 X {end1} > 255 X {start} - 255 * {overlap} / ? ?"
    else:  # constant merge with weight
        # Expression: if x < start_n → 0 elif x >= end_n → 255 else → mask_val
        expr = f"X {end} >= 255 X {start} < 0 {mask_val} ? ?"
    return mask.akarin.Expr(expr)


def _make_vertical_blend_mask_akarin(width: int, height: int, overlap: int, base_h: int, weight: float) -> vs.VideoNode:
    """Build a vertical blend mask using akarin.Expr for tile seam blending.

    Analogous to _make_horizontal_blend_mask_akarin but operates on the Y axis.

    :param width:   Mask width in pixels.
    :param height:  Mask height in pixels (= 2 × base_h).
    :param overlap: Half-height of the blending transition zone.
    :param base_h:  Base tile height (centre of transition at y == base_h).
    :param weight:  Blend weight in the transition zone [0, 1]. 0 = linear ramp.
    :return:        GRAY8 mask clip.
    """
    mask = core.std.BlankClip(width=width, height=height, format=vs.GRAY8)

    mask_val = int(round(weight * 255))
    start = base_h - overlap
    start1 = start + 1
    end = base_h + overlap
    end1 = end - 1

    if mask_val == 0:  # linear merge
        expr = f"Y {start1} < 0 Y {end1} > 255 Y {start} - 255 * {overlap} / ? ?"
    else:  # constant merge with weight
        expr = f"Y {end} >= 255 Y {start} < 0 {mask_val} ? ?"
    return mask.akarin.Expr(expr)


def _blend_horizontal(left: vs.VideoNode, right: vs.VideoNode, overlap: int, base_w: int, weight: float) -> vs.VideoNode:
    """Blend two horizontally adjacent tiles across their shared overlap zone.

    Both tiles are padded to 2×base_w width and merged with a horizontal blend mask.
    Falls back to std.StackHorizontal when overlap==0.

    :param left:    Left tile clip.
    :param right:   Right tile clip.
    :param overlap: Overlap width in pixels.
    :param base_w:  Base tile width (canvas = 2×base_w).
    :param weight:  Blend weight; 0 = linear ramp, >0 = constant weight.
    :return:        Merged clip of width 2×base_w.
    """
    if overlap <= 0:
        return core.std.StackHorizontal([left, right])

    out_w = base_w * 2
    # Pad to out_w, keep original height
    left_padded = core.std.AddBorders(left, right=out_w - left.width, top=0, bottom=0, left=0)
    right_padded = core.std.AddBorders(right, left=out_w - right.width, top=0, bottom=0, right=0)

    #img1 = frame_to_image(left_padded.get_frame(0))
    #img2 = frame_to_image(right_padded.get_frame(0))

    h = left_padded.height  # must match right_padded.height
    mask = _make_horizontal_blend_mask_akarin(out_w, h, overlap, base_w, weight)

    #mask_rgb = vs.core.resize.Bicubic(mask, format=vs.RGB24, range_s="full")

    clip_merged = core.std.MaskedMerge(left_padded, right_padded, mask)

    return clip_merged

def _blend_vertical(top: vs.VideoNode, bottom: vs.VideoNode, overlap: int, base_h: int, weight: float) -> vs.VideoNode:
    """Blend two vertically adjacent tiles across their shared overlap zone.

    Both tiles are padded to 2×base_h height and merged with a vertical blend mask.
    Falls back to std.StackVertical when overlap==0.

    :param top:     Top tile clip.
    :param bottom:  Bottom tile clip.
    :param overlap: Overlap height in pixels.
    :param base_h:  Base tile height (canvas = 2×base_h).
    :param weight:  Blend weight; 0 = linear ramp, >0 = constant weight.
    :return:        Merged clip of height 2×base_h.
    """
    if overlap <= 0:
        return core.std.StackVertical([top, bottom])

    out_h = base_h * 2
    top_padded = core.std.AddBorders(top, bottom=out_h - top.height, left=0, right=0, top=0)
    bottom_padded = core.std.AddBorders(bottom, top=out_h - bottom.height, left=0, right=0, bottom=0)

    w = top_padded.width
    mask = _make_vertical_blend_mask_akarin(w, out_h, overlap, base_h, weight)

    clip_merged = core.std.MaskedMerge(top_padded, bottom_padded, mask)

    return clip_merged
