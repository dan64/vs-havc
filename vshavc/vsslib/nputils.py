"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-02-29
version: 
LastEditors: Dan64
LastEditTime: 2026-05-17
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Library of Numpy utlity functions.
"""
import numpy as np
import cv2

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
implementation of max() function on numpy array, beacuse this function 
is not available on the base library.  
"""


def array_max(a: np.ndarray, a_max: any, dtype: np.dtype = np.uint8) -> np.ndarray:
    """
    Function to cap the values of np matrix to a_max

    Args:
        a : np matrix
        a_max: max allowed value for the matrix elements
        dtype: np return type (default: np.uint8)
    Return:
        np.ndarray : matrix with values capped to a_max
    """
    return np.where(a > a_max, a_max, a).astype(dtype)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
implementation of min() function on numpy array, beacuse this function 
is not available on the base library.  
"""


def array_min(a: np.ndarray, a_min: any, dtype: np.dtype = np.uint8) -> np.ndarray:
    """
    Function to floor the values of np matrix to a_min

    Args:
        a : np matrix
        a_min: min allowed value for the matrix elements
        dtype: np return type (default: np.uint8)
    Return:
        np.ndarray : matrix with values floored to a_min
    """
    return np.where(a < a_min, a_min, a).astype(dtype)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
implementation of min(max()) function on numpy array.
"""

def array_clip(a: np.ndarray, a_min: any, a_max: any, dtype: np.dtype = np.uint8) -> np.ndarray:
    """
    Function to clip the values of np matrix from a_min to a_max

    Args:
       a : np matrix
       a_max: max allowed value for the matrix elements
       a_min: min allowed value for the matrix elements
       dtype: np return type (default: np.uint8)
    Return:
       np.ndarray : matrix.clip(a_min, a_max)
    """
    a_m = array_max(a, a_max, dtype)
    return array_min(a_m, a_min, dtype)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
convert an NP image to gray or B&W if threshold > 0 
"""


def np_rgb_to_gray(img_np: np.ndarray, threshold: float = 0) -> np.ndarray:
    """Convert an RGB numpy array to grayscale (or binary B&W if threshold > 0).

    Uses standard luma weights (0.299R + 0.587G + 0.114B). If threshold > 0
    each channel is set to 255 where luma > threshold*255, else 0.

    :param img_np:    Input RGB array, shape (H, W, 3), dtype uint8.
    :param threshold: Binary threshold in [0, 1]. 0 = grayscale, >0 = B&W.
    :return:          Grayscale or binary RGB array with the same shape.
    """
    R = img_np[:, :, 0]
    G = img_np[:, :, 1]
    B = img_np[:, :, 2]

    R = R * 0.299
    G = G * 0.587
    B = B * 0.114

    tresh = round(threshold * 255)

    luma_np = R + G + B
    luma_np = luma_np.clip(0, 255)

    gray_np = img_np.copy()

    for i in range(3):
        if threshold > 0:
            gray_np[:, :, i] = np.where(luma_np > tresh, 255, 0)
        else:
            gray_np[:, :, i] = luma_np

    return gray_np


def np_get_luma(img_np: np.ndarray) -> np.ndarray:
    """Extract the luma (Y) channel from an RGB numpy array.

    Applies standard luma weights (0.299R + 0.587G + 0.114B).

    :param img_np: Input RGB array, shape (H, W, 3), dtype uint8.
    :return:       2-D luma array, shape (H, W), dtype float, range [0, 255].
    """
    R = img_np[:, :, 0]
    G = img_np[:, :, 1]
    B = img_np[:, :, 2]

    R = R * 0.299
    G = G * 0.587
    B = B * 0.114

    luma_np = R + G + B
    luma_np = luma_np.clip(0, 255)

    return luma_np


def w_np_rgb_to_gray(img_np: np.ndarray, dark_luma: float = 0, luma_white: float = 0.90,
                     as_weight: bool = True) -> np.ndarray:
    """Convert an RGB array to a per-pixel weight map with a gradient in the luma range.

    When dark_luma > 0 a linear gradient is computed from 0 at dark_luma to 1 at
    luma_white. When dark_luma == 0 the result is the normalised luma (range [0, 1] if
    as_weight=True, else [0, 255]).

    :param img_np:     Input RGB array, shape (H, W, 3), dtype uint8.
    :param dark_luma:  Lower boundary of the gradient ramp (fraction, [0, 1]).
    :param luma_white: Upper boundary of the gradient ramp (fraction, [0, 1]).
    :param as_weight:  If True return float values in [0, 1]; else uint8 [0, 255].
    :return:           Weight array, shape (H, W, 3).
    """
    R = img_np[:, :, 0]
    G = img_np[:, :, 1]
    B = img_np[:, :, 2]

    R = R * 0.299
    G = G * 0.587
    B = B * 0.114

    luma_np = R + G + B
    luma_np = luma_np.clip(0, 255)

    gray_np = img_np.copy()

    if dark_luma > 0:
        gray_np = gray_np.astype(float)
        max_white = round(luma_white * 255)

        tresh = min(round(dark_luma * 255), max_white - 10)

        grad = round(1 / (max_white - tresh), 3)

        luma_grad = ((luma_np - tresh) * grad).astype(float)

        weighted_luma = array_clip(luma_grad, 0.0, 1.0, np.float32)

        if as_weight:
            gray_np = gray_np.astype(float)
        else:
            weighted_luma = np.multiply(weighted_luma, 255).clip(0, 255).astype(np.uint8)

        for i in range(3):
            gray_np[:, :, i] = weighted_luma

    else:
        if as_weight:
            gray_np = gray_np.astype(float)
            luma_np = np.divide(luma_np, 255.0)
        for i in range(3):
            gray_np[:, :, i] = luma_np

    return gray_np


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
merge image1 with image2 using the mask (white->img2, black->img1) 
"""


def np_image_mask_merge(img1_np: np.ndarray, img2_np: np.ndarray,
                        mask_np: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Merge two RGB arrays using a binary mask (white → img2, black → img1).

    :param img1_np:   Base image array, shape (H, W, 3), dtype uint8.
    :param img2_np:   Overlay image array, same shape as img1_np.
    :param mask_np:   Mask array, same shape. White (255 or 1) selects img2.
    :param normalize: If True, divide mask by 255 before blending.
    :return:          Merged array, dtype uint8.
    """
    if normalize:
        mask_white = (mask_np / 255).astype(float)  # pass only white
        mask_black = (1 - mask_white).astype(float)  # pass only black
    else:
        mask_white = mask_np.astype(float)
        mask_black = (1 - mask_white).astype(float)

    img_np = img1_np.copy()

    img_m = img1_np * mask_black + img2_np * mask_white

    for i in range(3):
        img_np[:, :, i] = img_m[:, :, i].clip(0, 255).astype(np.uint8)

    return img_np


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
numpy weighted merge of image1 with image2 using the mask (white->img2, black->img1) 
"""


def w_np_image_mask_merge(img1_np: np.ndarray, img2_np: np.ndarray,
                          mask_w_np: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    numpy weighted merge of image1 with image2 using the mask (black->img1, white->img2)

    Args:
       img1_np: np.ndarray
       img2_np: np.ndarray
       mask_w_np: mask (w=0 -> black, w=1 -> white, gray in between)
       normalize: pixel will be divided by 255
    Return:
       img1*(1-w) + img2*w (if w=0 return img1, if w=1 return img2)
    """

    if normalize:
        mask_white = (mask_w_np / 255).astype(float)  # pass only white
        mask_black = (1 - mask_white).astype(float)  # pass only black
    else:
        mask_white = mask_w_np.astype(float)
        mask_black = (1 - mask_white).astype(float)

    img_np = img1_np.copy()

    img_m = (np.multiply(img1_np, mask_black) + np.multiply(img2_np, mask_white))

    for i in range(3):
        img_np[:, :, i] = img_m[:, :, i].clip(0, 255).astype(np.uint8)

    return img_np


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
"""


def np_weighted_merge(img1_np: np.ndarray, img2_np: np.ndarray, weight: float = 0.5) -> np.ndarray:
    """
    numpy implementation of image merge on 3 planes, faster than vs.core.std.Merge()

    Args:
       img1_np: np.ndarray
       img2_np: np.ndarray
       weight: float = 0.5
    Return:
       img1*(1-w) + img2*w (if w=0 return img1, if w=1 return img2)
    """
    img_new = np.copy(img1_np)

    img_m = (np.multiply(img1_np, 1 - weight) + np.multiply(img2_np, weight)).clip(0, 255).astype(np.uint8)
    img_new[:, :, 0] = img_m[:, :, 0]
    img_new[:, :, 1] = img_m[:, :, 1]
    img_new[:, :, 2] = img_m[:, :, 2]

    return img_new

def np_luma_blend(img_np: np.ndarray, img_new_np: np.ndarray, f_luma: float = 0.5, luma_limit: float = 0.6,
                     alpha: float = 0.95, min_w: float = 0.10, decay: float = 2.0) -> np.ndarray:
    """Blend two images with a weight that decreases when the frame is dark.

    When f_luma < luma_limit the blend weight is
    ``w = max(alpha * (f_luma / luma_limit)^decay, min_w)``.
    When f_luma >= luma_limit img_new is returned unchanged.

    :param img_np:     Original image array (H, W, 3), uint8.
    :param img_new_np: New (colorized) image array, same shape.
    :param f_luma:     Average luma of the frame, range [0, 1].
    :param luma_limit: Luma threshold below which blending activates.
    :param alpha:      Maximum blend weight assigned to img_new.
    :param min_w:      Minimum blend weight for very dark frames.
    :param decay:      Power-law exponent controlling how quickly weight drops.
    :return:           Blended array, dtype uint8.
    """
    # Luma merge
    if f_luma < luma_limit:
        bright_scale = pow(f_luma / luma_limit, decay)
        w = max(alpha * bright_scale, min_w)
        # img_m = img * (1.0 - w) + img_new * w
        img_m = np_weighted_merge(img_np, img_new_np, w)
    else:
        img_m = img_new_np

    return img_m


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function to copy the chroma parametrs "U", "V", of "img_m" in "orig" 
"""


def chroma_np_post_process(img_np: np.ndarray, orig_np: np.ndarray) -> np.ndarray:
    """Copy the chroma (U, V) planes of img_np into orig_np, keeping orig_np's luma.

    :param img_np:  Source of chroma (U, V), RGB array (H, W, 3), uint8.
    :param orig_np: Source of luma (Y), RGB array, same shape.
    :return:        RGB array with Y from orig_np and U/V from img_np.
    """
    img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    # copy the chroma parametrs "U", "V", of "img_m" in "orig" 
    orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)
    orig_copy = np.copy(orig_yuv)
    orig_copy[:, :, 1:3] = img_yuv[:, :, 1:3]
    return cv2.cvtColor(orig_copy, cv2.COLOR_YUV2RGB)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Function to add hue correction in cv2 HSV color space. 
hue range [-360.+360], converted to [-180.+180] 
"""


def np_hue_add(hsv_s: np.ndarray = None, hue: float = 0):
    """Shift the hue channel of an OpenCV HSV array in-place.

    The input hue range is [-360, +360] degrees; it is halved internally to
    match OpenCV's [0, 180] Hue convention. Wraps around the boundaries.

    :param hsv_s: Hue channel array (H, W), float, range [0, 180].
    :param hue:   Hue shift in degrees, range [-360, +360].
    :return:      Shifted hue array, same shape and dtype.
    """
    if hue == 0:
        return hsv_s

    hue_half = 0.5 * min(max(int(hue), -360), 360)

    hsv_s = hsv_s + hue_half
    hsv_s = np.where(hsv_s > 180, hsv_s - 180, hsv_s)
    hsv_s = np.where(hsv_s < 0, hsv_s + 180, hsv_s)

    return hsv_s


def np_image_gamma_contrast(np_img: np.ndarray = None, gamma: float = 1.0, cont: float = 1.0, perc: float = 5):
    """Apply gamma correction and contrast stretch to an RGB numpy array.

    Contrast is computed by percentile-clipping the Y channel to [perc, 100-perc]
    and rescaling to [0, 1] before multiplying by cont. Gamma is then applied
    as Y_out = (Y/255)^(1/gamma) * 255.

    :param np_img: Input RGB array (H, W, 3), uint8.
    :param gamma:  Gamma exponent (> 1 brightens, < 1 darkens). 1.0 = no-op.
    :param cont:   Contrast factor applied after percentile normalisation. 1.0 = no-op.
    :param perc:   Percentile used for contrast clipping, range [0, 50].
    :return:       Adjusted RGB array, uint8.
    """
    if cont == 1.0 and gamma == 1.0:
        return np_img

    yuv = cv2.cvtColor(np_img, cv2.COLOR_RGB2YUV)

    y = yuv[:, :, 0]
    yuv_new = np.copy(yuv)

    if cont != 1:
        y_min = np.percentile(y, perc)
        y_max = np.percentile(y, 100 - perc)
        y_fix = np.clip(y, y_min, y_max)
        y_cont = ((y_fix - y_min) * cont / (y_max - y_min))

        y_cont = array_clip(y_cont, 0, 1, np.float32) * 255

        y_new = y_cont.clip(0, 255).astype(int)
    else:
        y_new = y

    if gamma != 1:
        y_new = np.power(y_new / 255, 1 / gamma)
        y_new = np.multiply(y_new, 255).clip(0, 255).astype(np.uint8)

    yuv_new[:, :, 0] = y_new

    np_img_rgb = cv2.cvtColor(yuv_new, cv2.COLOR_YUV2RGB)

    return np_img_rgb


def isfloat(x) -> bool:
    """Return True if x can be converted to a float, False otherwise."""
    try:
        n = float(x)
        return True
    except ValueError:
        return False
