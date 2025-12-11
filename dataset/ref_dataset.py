# -*- coding: utf-8 -*-
# Directory layout (data_root):
#   Camo/test/Imgs/<Class>/<name>.jpg(.png/.jpeg)
#   Camo/test/GT/<Class>/<name>.png

import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# Keep type list consistent with dev (order matters)
TYPE_LIST = ['Shrimp', 'Duck', 'Moth', 'Chameleon', 'BatFish', 'Worm',
             'Turtle', 'ClownFish', 'LeafySeaDragon', 'Snake', 'Gecko',
             'Mockingbird', 'Beetle', 'Butterfly', 'Bug', 'Caterpillar',
             'Toad', 'FrogFish', 'Spider', 'Dragonfly', 'Katydid', 'Sheep',
             'Bird', 'Reccoon', 'Grouse', 'Deer', 'Octopus', 'Tiger', 'Fish',
             'Cicada', 'Flounder', 'Rabbit', 'Frog', 'Lion', 'StarFish',
             'Grasshopper', 'Bee', 'Frogmouth', 'Wolf', 'Lizard', 'Heron', 'Bat',
             'Kangaroo', 'Mantis', 'Monkey', 'Owl', 'Pipefish', 'Human', 'Dog',
             'Ant', 'ScorpionFish', 'Cheetah', 'Centipede', 'Sciuridae',
             'Crocodile', 'Leopard', 'Slug', 'SeaHorse', 'Crab', 'Giraffe', 'Cat',
             'GhostPipefish', 'StickInsect', 'Bittern'
             ]
TYPE_MAP = {t: i for i, t in enumerate(TYPE_LIST)}


def _find_mask_for_image(img_path: Path, gt_root: Path) -> Path:
    # Expect same class directory and stem
    cate = img_path.parent.name
    stem = img_path.stem
    # Prefer .png as in dev
    png_path = gt_root / cate / f"{stem}.png"
    if png_path.exists():
        return png_path
    # Fallback .jpg
    jpg_path = gt_root / cate / f"{stem}.jpg"
    if jpg_path.exists():
        return jpg_path
    raise FileNotFoundError(f"GT not found for image: {img_path}")


def _imread_rgb(path: str) -> np.ndarray:
    """Read image by OpenCV and convert to RGB float32 in [0,1]."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    assert bgr is not None, f"Image Not Found: {path}"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    return rgb


def _normalize_imagenet(img_rgb01: np.ndarray) -> np.ndarray:
    """Apply ImageNet mean/std normalization on RGB image in [0,1]."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,1,3)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,1,3)
    return (img_rgb01 - mean) / std


def _ms_resize(img: np.ndarray, base_h: int, base_w: int, scales=(1.5, 1.0, 0.5)) -> dict:
    """Multi-scale resize using cv2.INTER_LINEAR. Returns dict with keys '1.5','1.0','0.5'."""
    out = {}
    for s in scales:
        th = int(round(base_h * s))
        tw = int(round(base_w * s))
        resized = cv2.resize(img, dsize=(tw, th), interpolation=cv2.INTER_LINEAR)
        out[f"{s}"] = resized
    return out


class R2CTestDataset(Dataset):
    """
    Returns:
      dict(
        data = {
          "image1.5": Tensor [3,Hl,Wl],
          "image1.0": Tensor [3,Hm,Wm],
          "image0.5": Tensor [3,Hs,Ws],
          "img_type": LongTensor (index)
        },
        info = {
          "mask_path": str, "group_name": "image", "class_name": str, "img_path": str
        }
      )
    """
    def __init__(self, data_root: str, input_size: int = 384, exts: Tuple[str, ...] = (".jpg", ".png", ".jpeg")):
        super().__init__()
        self.data_root = Path(data_root)
        self.img_root = self.data_root / "Camo" / "test" / "Imgs"
        self.gt_root = self.data_root / "Camo" / "test" / "GT"
        assert self.img_root.exists(), f"Not found: {self.img_root}"
        assert self.gt_root.exists(), f"Not found: {self.gt_root}"

        # Collect image-mask pairs
        self.samples: List[Tuple[Path, Path]] = []
        for cate_dir in sorted(self.img_root.iterdir()):
            if not cate_dir.is_dir():
                continue
            for p in sorted(cate_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in exts:
                    try:
                        m = _find_mask_for_image(p, self.gt_root)
                        self.samples.append((p, m))
                    except FileNotFoundError:
                        pass

        # Base size for multi-scale
        self.base_h = int(input_size)
        self.base_w = int(input_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, mask_path = self.samples[index]
        class_name = img_path.parent.name
        if class_name not in TYPE_MAP:
            raise KeyError(f"Class '{class_name}' not in TYPE_LIST. Please update TYPE_LIST accordingly.")
        img_type_idx = TYPE_MAP[class_name]

        # Read and preprocess
        img = _imread_rgb(str(img_path))        # H,W,3 in [0,1], float32
        img = _normalize_imagenet(img)          # normalized

        # Multi-scale resize (1.5x, 1.0x, 0.5x) using cv2
        ms = _ms_resize(img, base_h=self.base_h, base_w=self.base_w, scales=(1.5, 1.0, 0.5))

        # To torch tensors [C,H,W]
        image15 = torch.from_numpy(ms["1.5"].transpose(2, 0, 1))
        image10 = torch.from_numpy(ms["1.0"].transpose(2, 0, 1))
        image05 = torch.from_numpy(ms["0.5"].transpose(2, 0, 1))

        data = dict(
            **{"image1.5": image15, "image1.0": image10, "image0.5": image05},
            img_type=torch.tensor(img_type_idx, dtype=torch.long),
        )
        info = dict(mask_path=str(mask_path), group_name="image", class_name=class_name, img_path=str(img_path))
        return dict(data=data, info=info)