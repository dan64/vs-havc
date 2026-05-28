"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2025-02-21
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Library of functions used by "HAVC" to restore color and change the hue of frames.
"""

import numpy as np
import cv2
from PIL import Image
import vapoursynth as vs
from vsdeoldify.vsslib.nputils import np_image_mask_merge, np_weighted_merge, np_hue_add, w_np_image_mask_merge, isfloat

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Restore the colors of past/future frame. The restore is applied using a mask
to select only the gray images on HSV color space.
The ranges that OpenCV manage for HSV format are the following:
- Hue range is [-180,+180], 
- Saturation range is [0,255] 
- Value range is [0,255].
For the 8-bit images, H is converted to H/2 to fit to the [0,255] range. 
So the range of hue in the HSV color space of OpenCV is [0,179]
The vector order is: H = 0, S = 1, V = 2
"""


def restore_color(img_color: Image = None, img_gray: Image = None, sat: float = 1.0, tht: int = 15, weight: float = 0,
                  tht_scen: float = 0.8, hue_adjust: str = 'none', return_mask: bool = False) -> Image:
    """Restore the colours of gray pixels in img_gray using a binary HSV saturation mask.

    Pixels in img_gray with HSV-S < tht are considered gray and replaced with
    (optionally desaturated) pixels from img_color. If the frame is nearly fully gray
    (ratio > tht_scen) it is returned unchanged.

    :param img_color:   Source of replacement colours (PIL RGB image).
    :param img_gray:    Target image with gray areas to be coloured (PIL RGB image).
    :param sat:         Saturation multiplier applied to img_color before merging. Default 1.0.
    :param tht:         Saturation threshold to identify gray pixels [0, 255]. Default 15.
    :param weight:      Post-merge blend weight: >0 blends back with img_gray, <0 with img_color. Default 0.
    :param tht_scen:    If gray-pixel ratio exceeds this value the frame is returned unchanged. Default 0.8.
    :param hue_adjust:  Chroma adjustment string applied after restoration. 'none' = disabled.
    :param return_mask: If True, return the binary gray-pixel mask instead of the restored image.
    :return:            Colour-restored PIL RGB image (or mask if return_mask=True).
    """
    np_color = np.asarray(img_color)
    np_gray = np.asarray(img_gray)

    hsv_color = cv2.cvtColor(np_color, cv2.COLOR_RGB2HSV)
    hsv_gray = cv2.cvtColor(np_gray, cv2.COLOR_RGB2HSV)

    # desatured the color image
    hsv_color[:, :, 1] = hsv_color[:, :, 1] * min(max(sat, 0), 10)

    np_color_sat = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)

    hsv_s = hsv_gray[:, :, 1]

    hsv_mask = np.where(hsv_s < tht, 255, 0)  # white only gray pixels

    scenechange = np.mean(hsv_mask) / 255

    if 0 < tht_scen < 1 and scenechange > tht_scen:
        if hue_adjust != "" and hue_adjust != "none":
            return adjust_hue_range(img_gray, hue_adjust=hue_adjust)
        else:
            return img_gray

    mask_rgb = np_gray.copy()

    for i in range(3):
        mask_rgb[:, :, i] = hsv_mask

    if return_mask:
        return Image.fromarray(mask_rgb, 'RGB').convert('RGB')

    np_restored = np_image_mask_merge(np_gray, np_color_sat, mask_rgb)

    if weight > 0:
        np_restored = np_weighted_merge(np_restored, np_gray, weight)  # merge with gray frame
    if weight < 0:
        np_restored = np_weighted_merge(np_restored, np_color_sat, -weight)  # merge with colored frame

    img_restored = Image.fromarray(np_restored, 'RGB').convert('RGB')

    if hue_adjust != "" and hue_adjust != "none":
        return adjust_hue_range(img_restored, hue_adjust=hue_adjust)
    else:
        return img_restored


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Restore the gray frame colors frame. The restore is applied using a gradient mask
to select only the gray images on HSV color space.
The vector order is: H = 0, S = 1, V = 2
"""


def restore_color_gradient(img_color: Image = None, img_gray: Image = None, sat: float = 1.0, tht: int = 50,
                           weight: float = 0, alpha: float = 2.0, return_mask: bool = False, algo: int = 0) -> Image:
    """Restore colours of gray pixels in img_gray using a smooth gradient mask.

    Unlike restore_color (binary mask) this function builds a gradient mask from the
    saturation channel so the transition from gray to coloured areas is smooth.

    :param img_color:   Source of replacement colours (PIL RGB image).
    :param img_gray:    Target image with gray areas (PIL RGB image).
    :param sat:         Saturation multiplier applied to img_color. Default 1.0.
    :param tht:         Saturation threshold for the gradient [0, 255]. Default 50.
    :param weight:      Post-merge blend weight: >0 toward img_color, <0 toward img_gray. Default 0.
    :param alpha:       Steepness of the gradient curve. Higher = more aggressive. Default 2.0.
    :param return_mask: If True, return the gradient mask instead of the restored image.
    :param algo:        Mask algorithm: 0 = linear-steep (default), 1 = linear, 2 = exponential.
    :return:            Colour-restored PIL RGB image (or mask if return_mask=True).
    """
    np_color = np.asarray(img_color)
    np_gray = np.asarray(img_gray)

    hsv_color = cv2.cvtColor(np_color, cv2.COLOR_RGB2HSV)
    hsv_gray = cv2.cvtColor(np_gray, cv2.COLOR_RGB2HSV)

    # desatured the color image
    if sat != 1.0:
        hsv_color[:, :, 1] = hsv_color[:, :, 1] * min(max(sat, 0), 10)

    np_color_sat = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)

    hsv_s = hsv_gray[:, :, 1]

    hsv_mask = w_np_gradient_mask(hsv_s, tht, alpha, algo)  # white only gray pixels

    mask_rgb = np_gray.copy()

    for i in range(3):
        mask_rgb[:, :, i] = hsv_mask

    if return_mask:
        return Image.fromarray(mask_rgb, 'RGB').convert('RGB')

    np_restored = w_np_image_mask_merge(np_gray, np_color_sat, mask_rgb, normalize=True)

    if weight > 0:
        np_restored = np_weighted_merge(np_restored, np_color_sat, weight)  # merge with colored frame

    if weight < 0:
        np_restored = np_weighted_merge(np_restored, np_gray, -weight)  # merge with gray frame

    img_restored = Image.fromarray(np_restored, 'RGB').convert('RGB')

    return img_restored


def w_np_gradient_mask_steep(img_np: np.ndarray, tht: int = 15, alpha: float = 2.0, steep: float = 2.0) -> np.ndarray:
    """Build a steep linear gradient mask from a saturation channel array.

    Returns 255 for fully gray pixels (S near 0) decaying to 0 around S=tht.

    :param img_np:  HSV saturation channel, shape (H, W), range [0, 255].
    :param tht:     Saturation threshold; pixels below are considered gray. Default 15.
    :param alpha:   Controls the slope steepness on the upper side of the gradient. Default 2.0.
    :param steep:   Multiplier controlling the lower-slope behaviour. Default 2.0.
    :return:        Mask array, shape (H, W), values in [0, 255] as int.
    """
    luma_np = img_np.clip(0, 255)

    # grad = np.where(luma_np < tht, luma_np, tht + (luma_np - tht)*alpha)
    # luma_grad = (255.0 - luma_np - grad).clip(0, 255).astype(int)

    grad = np.where(luma_np < tht, steep*luma_np/alpha - tht, steep*(luma_np - tht)*alpha)
    luma_grad = (255.0 - tht - grad).clip(0, 255).astype(int)

    return luma_grad


def w_np_gradient_mask(saturation: np.ndarray, tht: int = 15, alpha: float = 2.0, algo: int = 0) -> np.ndarray:
    """
    Create a mask that is WHITE where saturation is LOW (gray areas).
    Mask is 255 at S=0, 128 at S=tht, 0 at S=2*tht.

    Args:
        saturation: HSV S channel (0-255)
        tht: threshold — pixels with S <= tht are considered "gray"
        alpha: smoothness (higher = softer transition)
        algo: algorithm to build the mask, allowed values are:
                [0] = Linear decay with steep gradient,
                [1] = Linear decay
                [2] = Exponential decay

    Returns:
        mask: 0-255, where 255 = fully gray (needs colorization)
    """

    if algo == 0:
        return w_np_gradient_mask_steep(saturation, tht, alpha)

    s = saturation.astype(np.float32)

    # Ensure tht is in valid range
    tht = int(np.clip(tht, 0, 255))

    if tht == 0:
        return np.zeros_like(saturation, dtype=np.uint8)

    if algo == 1:
        # Define max saturation for full falloff
        # We want falloff from S=0 to S=max_s, where max_s = min(2*tht, 255)
        max_s = min(2 * tht, 200)  # safe bound to 200
        s_clipped = np.clip(s, 0, max_s)

        # Linear mapping: S=0 → 1.0, S=max_s → 0.0
        # Apply power law for nonlinear falloff
        mask_norm = (1.0 - (s_clipped / max_s)) ** alpha
    else:
        # Normalize S to [0, 1] relative to tht
        s_rel = np.clip(s / tht, 0, 2)  # cap at 2x tht

        # Exponential decay: mask = exp(-alpha * s_rel * ln(2))
        # So that at s_rel = 1 (S = tht), mask = exp(-ln(2)) = 0.5
        mask_norm = np.exp(-alpha * s_rel * np.log(2))

        # Cap at S = 2*tht (optional)
        mask_norm = np.where(s >= 2 * tht, 0.0, mask_norm)

    # Scale to [0, 255]
    mask =  (np.clip(mask_norm * 255, 0, 255)).astype(np.uint8)

    return mask

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Change a given range of colors in HSV color space. 
The range is defined by the hue values in degree (range: 0-360)
In OpenCV, for the 8-bit images, H is converted to H/2 to fit to the [0,255] range. 
So the range of hue in the HSV color space of OpenCV is [0,179].
hue_range syntax: "hue1_min:hue1_max,..,hueN_min,hueN_max|adjust, weight"
where:
adjust: if > 0 and < 10 -> saturation parameter else -> hue_shift
weight: if > 0 -> merge with desaturared frame, if < 0 -> merge with colored orginal frame
"""


def adjust_hue_range(img_color: Image = None, hue_adjust: str = 'none', return_mask: bool = False) -> Image:
    """Adjust saturation and/or hue in a specified colour range of a PIL image.

    Parses hue_adjust (format: "hue_range|sat_or_hue,weight") and delegates to
    adjust_chroma. Returns the original image when hue_adjust is 'none' or empty.

    :param img_color:   Input PIL RGB image.
    :param hue_adjust:  Chroma adjustment string (e.g. "300:360|0.8,0.1"). 'none' = bypass.
    :param return_mask: If True, return the hue-range selection mask.
    :return:            Adjusted PIL RGB image (or mask if return_mask=True).
    """
    if hue_adjust == 'none' or hue_adjust == '':
        return img_color

    param = _parse_hue_adjust(hue_adjust)

    if param is None:
        return img_color

    hue_range = param[0]
    sat = param[1]
    hue = param[2]
    weight = param[3]

    return adjust_chroma(img_color=img_color, hue_range=hue_range, sat=sat, hue=hue, weight=weight,
                         return_mask=return_mask)


def adjust_chroma(img_color: Image = None, hue_range: str = 'none', sat: float = 0.3, hue: int = 0, weight: float = 0,
                  return_mask: bool = False) -> Image:
    """Apply saturation and hue adjustments only to pixels within a specified hue range.

    Pixels in the hue range are replaced by the adjusted version; others are unchanged.
    Optional blend weight merges the result back with the original.

    :param img_color:   Input PIL RGB image.
    :param hue_range:   Hue range string (e.g. "300:360,0:30" for multiple ranges). 'none' = bypass.
    :param sat:         Saturation multiplier for the selected hue range. Default 0.3.
    :param hue:         Hue shift in degrees for the selected range. Default 0.
    :param weight:      Post-merge blend weight: >0 blend with adjusted frame, <0 with original. Default 0.
    :param return_mask: If True, return the hue-range selection mask instead of adjusted image.
    :return:            Adjusted PIL RGB image (or mask if return_mask=True).
    """
    if hue_range == 'none' or hue_range == '':
        return img_color

    np_color = np.asarray(img_color)

    np_gray = np_color.copy()
    np_gray = cv2.cvtColor(np_gray, cv2.COLOR_RGB2HSV)

    hsv_color = cv2.cvtColor(np_color, cv2.COLOR_RGB2HSV)

    # apply hue correction, range [-180,+180]
    if hue != 0:
        np_gray[:, :, 0] = np_hue_add(np_gray[:, :, 0], hue)

    # desatured the color image
    if sat != 1:
        np_gray[:, :, 1] = np_gray[:, :, 1] * min(max(sat, 0), 10)

    np_gray_rgb = cv2.cvtColor(np_gray, cv2.COLOR_HSV2RGB)

    hsv_s = hsv_color[:, :, 0]

    cond = _build_hue_conditions(hsv_s, hue_range)

    hsv_mask = np.where(cond, 255, 0)  # white only gray pixels

    mask_rgb = np_color.copy()

    for i in range(3):
        mask_rgb[:, :, i] = hsv_mask

    if return_mask:
        return Image.fromarray(mask_rgb, 'RGB').convert('RGB')

    np_restored = np_image_mask_merge(np_color, np_gray_rgb, mask_rgb)

    if weight > 0:
        if hue == 0:
            np_restored = np_weighted_merge(np_restored, np_gray_rgb, weight)
        else:
            np_restored = np_weighted_merge(np_restored, np_color, weight)
    if weight < 0:   # use np_color instead of np_gray_rgb, is assumed that hue == 0 (no color mapping)
        np_restored = np_weighted_merge(np_restored, np_color, -weight)

    return Image.fromarray(np_restored, 'RGB').convert('RGB')


def np_image_chroma_tweak(img_color_rgb: np.ndarray, sat: float = 1, bright: float = 0, hue: int = 0,
                          hue_adjust: str = 'none') -> np.ndarray:
    """Adjust saturation, brightness, hue, and optionally a restricted hue range on a NumPy array.

    Adjustments are applied in HSV space. If hue_adjust is set, an additional targeted
    hue-range correction is applied after the global adjustments.

    :param img_color_rgb: Input RGB array (H, W, 3), uint8.
    :param sat:           Saturation multiplier [0, 10]. Default 1.
    :param bright:        Brightness offset for V channel. Default 0.
    :param hue:           Global hue rotation in degrees [-360, +360]. Default 0.
    :param hue_adjust:    Chroma adjustment string for a specific hue range. 'none' = disabled.
    :return:              Adjusted RGB array, uint8.
    """
    if sat == 1 and bright == 0 and hue == 0 and hue_adjust == 'none':
        return img_color_rgb  # non changes

    hsv = cv2.cvtColor(img_color_rgb, cv2.COLOR_RGB2HSV)

    hsv[:, :, 0] = np_hue_add(hsv[:, :, 0], hue)
    hsv[:, :, 1] = hsv[:, :, 1] * min(max(sat, 0), 10)
    hsv[:, :, 2] = hsv[:, :, 2] * min(max(1 + bright, 0), 10)

    np_color_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if hue_adjust == 'none' or hue_adjust == '':
        return np_color_rgb

    param = _parse_hue_adjust(hue_adjust)

    if param is None:
        return np_color_rgb

    hue_range = param[0]
    sat = param[1]
    hue = param[2]  # override hue with the new value
    weight = param[3]

    np_gray = np_color_rgb.copy()
    np_gray = cv2.cvtColor(np_gray, cv2.COLOR_RGB2HSV)

    hsv_color = hsv.copy()

    # apply hue correction, range [-180.+180], converted to [-90.+90]
    if hue != 0:
        np_gray[:, :, 0] = np_hue_add(np_gray[:, :, 0], hue)

    # desatured the color image
    if sat != 1:
        np_gray[:, :, 1] = np_gray[:, :, 1] * min(max(sat, 0), 10)

    np_gray_rgb = cv2.cvtColor(np_gray, cv2.COLOR_HSV2RGB)

    hsv_s = hsv_color[:, :, 0]

    cond = _build_hue_conditions(hsv_s, hue_range)

    hsv_mask = np.where(cond, 255, 0)  # white only gray pixels

    mask_rgb = img_color_rgb.copy()

    for i in range(3):
        mask_rgb[:, :, i] = hsv_mask

    np_restored = np_image_mask_merge(img_color_rgb, np_gray_rgb, mask_rgb)

    if weight > 0:
        if hue == 0:
            np_restored = np_weighted_merge(np_restored, np_gray_rgb, weight)
        else:
            np_restored = np_weighted_merge(np_restored, img_color_rgb, weight)
    if weight < 0:
        np_restored = np_weighted_merge(np_restored, img_color_rgb, -weight)

    return np_restored


def np_adjust_chroma2(np_color_rgb: np.ndarray, np_gray_rgb: np.ndarray, hue_range: str = 'none',
                      return_mask: bool = False) -> np.ndarray:
    """Select pixels in hue_range from np_color_rgb and blend them into np_gray_rgb.

    The hue mask is built from np_color_rgb's H channel; masked pixels are taken from
    np_gray_rgb, unmasked pixels from np_color_rgb. Returns np_gray_rgb unchanged when
    hue_range is 'none'.

    :param np_color_rgb:  Reference array whose hue defines the selection mask (H, W, 3), uint8.
    :param np_gray_rgb:   Target array whose pixels are inserted in the selected hue range.
    :param hue_range:     Hue range string (e.g. "300:360"). 'none' = bypass.
    :param return_mask:   If True, return the selection mask (H, W, 3), uint8.
    :return:              Merged RGB array, uint8 (or mask if return_mask=True).
    """
    if hue_range == 'none' or hue_range == '':
        return np_gray_rgb

    hsv_color = cv2.cvtColor(np_color_rgb, cv2.COLOR_RGB2HSV)
    hsv_s = hsv_color[:, :, 0]

    cond = _build_hue_conditions(hsv_s, hue_range)

    hsv_mask = np.where(cond, 255, 0)  # white only gray pixels

    mask_rgb = np_color_rgb.copy()

    for i in range(3):
        mask_rgb[:, :, i] = hsv_mask

    if return_mask:
        return mask_rgb  # Image.fromarray(mask_rgb, 'RGB').convert('RGB')

    np_restored = np_image_mask_merge(np_color_rgb, np_gray_rgb, mask_rgb)

    return np_restored


def _parse_hue_adjust(hue_adjust: str = 'none') -> ():
    """Parse a chroma adjustment string into (hue_range, sat, hue, weight).

    Format: "hue_range|adjust,weight"
    - adjust: if in (0, 10) interpreted as saturation, otherwise as hue shift (int).
    - weight: float blend weight.

    :param hue_adjust: Adjustment string. 'none' or '' returns None.
    :return:           Tuple (hue_range: str, sat: float, hue: int, weight: float) or None on error.
    """
    p = hue_adjust.split("|")

    sat = 1.0
    hue = 0
    weight = 0

    num = len(p)
    if num < 1 or num > 2:
        return None

    #pp = p[0].split(":")
    #if not pp[0].isnumeric() or not pp[1].isnumeric():
    #    return None

    hue_range = p[0]

    if num == 1:
        return hue_range, sat, hue, weight

    sw = p[1].split(",")
    if len(sw) != 2 or not isfloat(sw[0]) or not isfloat(sw[1]):
        return None

    if (sw[0])[0] in ('-', '+'):
        hue = int(sw[0])
    else:
        sat = float(sw[0])

    if sat > 10:  # fix wrong input
        hue = int(sat)
        sat = 1.0

    weight = float(sw[1])

    return hue_range, sat, hue, weight


def _build_hue_conditions(hsv_s: np.ndarray = None, hue_range: str = None) -> np.ndarray:
    """Build a boolean condition mask for pixels whose hue falls in the specified ranges.

    Supports multiple comma-separated hue ranges (e.g. "300:360,0:30"). Hue values are
    divided by 2 to match OpenCV's 8-bit HSV H range [0, 180].

    :param hsv_s:     HSV Hue channel array (H, W), values in [0, 180] (OpenCV convention).
    :param hue_range: Comma-separated hue range string (e.g. "300:360" or "red,blue").
    :return:          Boolean array (H, W), True where pixel hue is within any specified range.
    """
    h_range = hue_range.split(",")
    h_len = len(h_range)

    hue_min, hue_max = _parse_hue_range(h_range[0])
    # For the 8-bit images, H is converted to H/2 to fit to the [0,255] range.
    c1 = hsv_s > hue_min * 0.5
    c2 = hsv_s < hue_max * 0.5
    cond = (c1 & c2)

    for i in range(1, h_len):
        hue_min, hue_max = _parse_hue_range(h_range[i])
        c1 = hsv_s > hue_min * 0.5
        c2 = hsv_s < hue_max * 0.5
        cond |= (c1 & c2)

    return cond


def _parse_hue_range(hue_range: str = None) -> ():
    """Convert a hue range string to a (min, max) degree tuple.

    Accepts named colour strings (e.g. "red", "blue-green") or numeric ranges
    in the format "min:max" (degrees, 0–360).

    :param hue_range: Colour name or "min:max" string.
    :return:          Tuple (hue_min, hue_max) in degrees [0, 360].
    :raises vs.Error: If the name is unknown or format is invalid.
    """
    # For color increments, each block in a given "hue_range" represents a Hue change of 30.
    match hue_range:
        case "red":
            rng = (0, 30)
        case "orange":
            rng = (30, 60)
        case "yellow":
            rng = (60, 90)
        case "yellow-green":
            rng = (90, 120)
        case "green":
            rng = (120, 150)
        case "blue-green":
            rng = (150, 180)
        case "cyan":
            rng = (180, 210)
        case "blue":
            rng = (210, 240)
        case "blue-violet":
            rng = (240, 270)
        case "violet":
            rng = (270, 300)
        case "red-violet":
            rng = (300, 330)
        case "rose":
            rng = (330, 360)
        case _:
            p = hue_range.split(":")
            if len(p) == 2 and p[0].isnumeric() and p[1].isnumeric():
                rng = (float(p[0]), float(p[1]))
            else:
                raise vs.Error("HybridAVC: unknown hue name: " + hue_range)

    return rng


def get_color_tune(hue_name: str = None) -> str:
    """Return the hue range string for a named colour-tune preset.

    Maps HAVC ColorFix names (e.g. "magenta", "violet/red") to their corresponding
    "min:max" hue range strings used by adjust_chroma / adjust_hue_range.

    :param hue_name: Colour-tune name (case-sensitive, lower-case).
    :return:         Hue range string (e.g. "270:300").
    :raises vs.Error: If hue_name is not a recognised preset.
    """
    # For color increments, each block in a given "hue_range" represents a Hue change of 30.
    match hue_name:
        case "magenta":
            rng = "270:300"
        case "magenta/violet":
            rng = "270:330"
        case "violet":
            rng = "300:330"
        case "violet/red":
            rng = "300:360"
        case "blue/magenta":
            rng = "240:300"
        case "yellow":
            rng = "60:90"
        case "yellow/orange":
            rng = "30:90"
        case "yellow/green":
            rng = "60:120"
        case _:
            raise vs.Error("HybridAVC: unknown color tune: " + hue_name)

    return rng
