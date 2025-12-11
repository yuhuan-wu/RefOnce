# -*- coding: utf-8 -*-

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.ref_dataset import R2CTestDataset
from models.refonce import RefOnce
from metrics.metric_caller import CalTotalMetric
from utils.ops import read_gray_array, imresize, clip_to_normalize, minmax, save_array_as_image


def parse_args():
    p = argparse.ArgumentParser("Release testing for RefOnce")
    p.add_argument("--data-root", type=str, required=True, help="R2C7K root containing Camo/")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    p.add_argument("--batch-size", type=int, default=22)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--input-size", type=int, default=384, help="Base input size")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--save-dir", type=str, default="", help="Directory to save predictions; empty string disables saving")
    p.add_argument("--save-preds", type=lambda x: str(x).lower() in ("true", "1", "yes", "y"), default=False,
                   help="Whether to save prediction maps")
    p.add_argument("--to-minmax", type=lambda x: str(x).lower() in ("true", "1", "yes", "y"), default=False,
                   help="Min-max results before saving")
    return p.parse_args()


def flexible_load(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = None
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        if state is None:
            # maybe already a state_dict-like
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state = ckpt
    else:
        # Unexpected format; let it fail loudly
        state = ckpt

    if state is None:
        raise RuntimeError(f"Unsupported checkpoint format: keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    # Strip possible 'module.' prefix from DDP training
    def strip_prefix_if_present(state_dict, prefix="module."):
        return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items() }

    state = strip_prefix_if_present(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warn] Missing keys while loading: {len(missing)} e.g., {missing[:8]}")
    if unexpected:
        print(f"[Warn] Unexpected keys while loading: {len(unexpected)} e.g., {unexpected[:8]}")


def build_loader(args) -> DataLoader:
    dataset = R2CTestDataset(data_root=args.data_root, input_size=args.input_size)
    # Custom collate to keep 'info' as list while stacking images
    def collate_fn(batch: List[dict]):
        datas = [b["data"] for b in batch]
        infos = [b["info"] for b in batch]
        out_data = dict(
            **{
                "image1.5": torch.stack([d["image1.5"] for d in datas], 0),
                "image1.0": torch.stack([d["image1.0"] for d in datas], 0),
                "image0.5": torch.stack([d["image0.5"] for d in datas], 0),
            },
            img_type=torch.stack([d["img_type"] for d in datas], 0),
        )
        return dict(data=out_data, info=infos)
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )



@torch.no_grad()
def run_test(args):
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    loader = build_loader(args)
    try:
        ds_len = len(loader.dataset)  # type: ignore[arg-type]
    except TypeError:
        ds_len = "unknown"
    print(f"Testing dataset loaded: {ds_len} images")

    model = RefOnce()
    flexible_load(model, args.checkpoint)
    model.to(device)
    model.eval()

    # Sequential metric computation
    calc_global = CalTotalMetric()

    def _process_item(pred: np.ndarray, mask_path: str, cls_name: str):
        """
        Per-sample pipeline (sequential):
        - read GT
        - resize prediction
        - optional minmax
        - optional save
        Return (pred_u8, mask_array, mask_path) for metric.step
        """
        mask_array = read_gray_array(mask_path).astype(np.uint8)  # uint8 [H,W]
        H, W = mask_array.shape

        # Resize prediction to GT size
        pred_resized = imresize(pred, target_h=H, target_w=W, interp="linear")
        pred_resized = np.clip(pred_resized, 0, 1)

        if args.to_minmax:
            pred_resized = minmax(pred_resized)

        if args.save_dir and args.save_preds:
            subdir = os.path.join(args.save_dir, cls_name)
            save_array_as_image(pred_resized, save_name=os.path.basename(mask_path), save_dir=subdir, to_minmax=False)

        pred_u8 = (pred_resized * 255).astype(np.uint8)
        return pred_u8, mask_array, mask_path

    pbar = tqdm(total=len(loader), ncols=79, desc="[TE]")
    for batch in loader:
        data = batch["data"]
        info_list = batch["info"]  # list of dicts, len=B

        # Move tensors
        images = {
            "image1.5": data["image1.5"].to(device, non_blocking=True),
            "image1.0": data["image1.0"].to(device, non_blocking=True),
            "image0.5": data["image0.5"].to(device, non_blocking=True),
            "img_type": data["img_type"].to(device, non_blocking=True),
        }

        logits = model(data=images)  # [B,1,H,W]
        probs = logits.sigmoid().squeeze(1).cpu().numpy()  # [B,H,W], float in [0,1]

        # Sequential preparation for current batch
        for i in range(probs.shape[0]):
            mask_path = info_list[i]["mask_path"]
            cls_name = info_list[i]["class_name"]
            pred_u8, mask_array, mpath = _process_item(probs[i], mask_path, cls_name)
            calc_global.step(pred_u8, mask_array, mpath)

        pbar.update(1)
    pbar.close()

    # Report results from deterministic global calculator
    results = calc_global.get_results()
    print("Final results (Smeasure / maxEm / maxFm / MAE):")
    print(f"Smeasure: {results['Smeasure']}, maxEm: {results['maxEm']}, maxFm: {results['maxFm']}, MAE: {results['MAE']}")
    # Also print full dict
    print(results)


def main():
    args = parse_args()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    run_test(args)


if __name__ == "__main__":
    main()