# -*- coding: utf-8 -*-
from numbers import Number
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------- Array ops ----------
def minmax(data_array: np.ndarray, up_bound: float = None) -> np.ndarray:
    if up_bound is not None:
        data_array = data_array / up_bound
    max_value = data_array.max()
    min_value = data_array.min()
    if max_value != min_value:
        data_array = (data_array - min_value) / (max_value - min_value)
    return data_array


def clip_to_normalize(data_array: np.ndarray, clip_range: tuple = None) -> np.ndarray:
    if clip_range is None:
        return minmax(data_array)
    clip_range = sorted(clip_range)
    if len(clip_range) == 3:
        clip_min, clip_mid, clip_max = clip_range
        assert 0 <= clip_min < clip_mid < clip_max <= 1, clip_range
        lower_array = data_array[data_array < clip_mid]
        higher_array = data_array[data_array > clip_mid]
        if lower_array.size > 0:
            lower_array = np.clip(lower_array, a_min=clip_min, a_max=1)
            max_lower = lower_array.max()
            lower_array = minmax(lower_array) * max_lower
            data_array[data_array < clip_mid] = lower_array
        if higher_array.size > 0:
            higher_array = np.clip(higher_array, a_min=0, a_max=clip_max)
            min_lower = higher_array.min()
            higher_array = minmax(higher_array) * (1 - min_lower) + min_lower
            data_array[data_array > clip_mid] = higher_array
    elif len(clip_range) == 2:
        clip_min, clip_max = clip_range
        assert 0 <= clip_min < clip_max <= 1, clip_range
        if clip_min != 0 and clip_max != 1:
            data_array = np.clip(data_array, a_min=clip_min, a_max=clip_max)
        data_array = minmax(data_array)
    else:
        raise NotImplementedError
    return data_array


def imresize(image_array: np.ndarray, target_h, target_w, interp: str = "linear") -> np.ndarray:
    _interp_mapping = dict(
        linear=cv2.INTER_LINEAR,
        cubic=cv2.INTER_CUBIC,
        nearst=cv2.INTER_NEAREST,
    )
    assert interp in _interp_mapping, f"Only support interp: {list(_interp_mapping.keys())}"
    return cv2.resize(image_array, dsize=(target_w, target_h), interpolation=_interp_mapping[interp])


def save_array_as_image(data_array: np.ndarray, save_name: str, save_dir: str, to_minmax: bool = False):
    """
    Save ndarray (float in [0,1] or uint8) to image path.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    arr = data_array
    if arr.dtype != np.uint8:
        if arr.max() > 1:
            # Assume already [0,255] float
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            if to_minmax:
                arr = minmax(arr, up_bound=1.0)
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(save_path, arr)


# ---------- IO ----------
def read_gray_array(path: str, div_255: bool = False, to_normalize: bool = False, thr: float = -1, dtype=np.float32) -> np.ndarray:
    """
    Read a grayscale image and optionally normalize to [0,1] or binarize by threshold.
    """
    assert path.endswith(".jpg") or path.endswith(".png"), f"Unsupported image suffix: {path}"
    gray_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert gray_array is not None, f"Image Not Found: {path}"
    if div_255:
        gray_array = gray_array / 255
    if to_normalize:
        gray_array = minmax(gray_array, up_bound=255)
    if thr >= 0:
        gray_array = gray_array > thr
    return gray_array.astype(dtype)


# ---------- Tensor ops ----------
def cus_sample(
    feat: torch.Tensor,
    mode=None,
    factors=None,
    *,
    interpolation: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Resize feature tensor by size or scale using torch.nn.functional.interpolate
    :param feat: 4D tensor [N,C,H,W]
    :param mode: None | "size" | "scale"
    :param factors: (H,W) for size mode, float for scale mode
    """
    if mode is None:
        return feat
    else:
        if factors is None:
            raise ValueError(
                f"factors should be valid when mode is not None. feat.shape={feat.shape}, mode={mode}"
            )
    interp_cfg = {}
    if mode == "size":
        if isinstance(factors, Number):
            factors = (factors, factors)
        assert isinstance(factors, (list, tuple)) and len(factors) == 2
        factors = [int(x) for x in factors]
        if factors == list(feat.shape[2:]):
            return feat
        interp_cfg["size"] = factors
    elif mode == "scale":
        assert isinstance(factors, (int, float))
        if factors == 1:
            return feat
        recompute_scale_factor = None
        if isinstance(factors, float):
            recompute_scale_factor = False
        interp_cfg["scale_factor"] = factors
        interp_cfg["recompute_scale_factor"] = recompute_scale_factor
    else:
        raise NotImplementedError(f"mode can not be {mode}")

    if interpolation == "nearest":
        if align_corners is False:
            align_corners = None
        assert align_corners is None, (
            "align_corners option can only be set with modes: linear|bilinear|bicubic|trilinear"
        )
    return F.interpolate(feat, mode=interpolation, align_corners=align_corners, **interp_cfg)