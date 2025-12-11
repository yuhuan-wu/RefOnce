# RefOnce
RefOnce: Distilling References into a Prototype Memory for Referring Camouflaged Object Detection

## Requirements

- Python 3.8+
- PyTorch and TorchVision
- Other deps: timm==0.4.12, py_sod_metrics==1.2.4, opencv-python, pillow, tqdm

Install example (choose a proper CUDA or CPU wheel as needed):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm==0.4.12 py_sod_metrics==1.2.4 opencv-python pillow tqdm
```

## Get Started (prepare data)

- Dataset root should contain the following structure (R2C7K example, at least test set should exist):

```
dataset/
  R2C7K/
    Camo/
      test/
        Imgs/<Class>/<name>.jpg|.png
        GT/<Class>/<name>.png
    Ref/   # keep original structure if present; not strictly required for inference
```

- Weights: place the checkpoint (e.g., RefOnce.pth) anywhere and pass it via --checkpoint.

- Entry points and useful references:
  - Shell runner: [test.sh](test.sh:1)
  - Python entry: [test.py](test.py:1)
  - Model: [RefOnce.__init__()](models/refonce.py:247), [RefOnce.test_forward()](models/refonce.py:331)

## Inference

Shell script (recommended): 

```bash
bash test.sh 0   # optional GPU index, default 0;
```

The script calls [test.py](test.py:1) with defaults:
- data-root: ./dataset/R2C7K
- checkpoint: RefOnce.pth
- batch-size: 22
- save-dir: ./output/release/
- save-preds: False (metrics only)

Notes:
- By default, predictions are not saved; enable with --save-preds True and provide --save-dir.
- Metrics are reported by [CalTotalMetric.get_results()](metrics/metric_caller.py:25) and include Smeasure, wFmeasure, MAE, adpEm, meanEm, maxEm, adpFm, meanFm, maxFm.

## Citation

If this release helps your research, please cite it. Example BibTeX (replace with your official paper info):

```bibtex
@article{wu2025refonce,
  title={RefOnce: Distilling References into a Prototype Memory for Referring Camouflaged Object Detection},
  author={Wu, Yu-Huan and Zhu, Zi-Xuan and Wang, Yan and Zhen, Liangli and Fan, Deng-Ping},
  journal={arXiv preprint arXiv:2511.20989},
  year={2025}
}
```

## Acknowledgements

This project is based on [ZoomNet](https://github.com/lartpang/ZoomNet), [RefCOD](https://github.com/zhangxuying1004/RefCOD), and [PySODMetrics](https://github.com/lartpang/PySODMetrics). Thanks to the authors for their open-source contributions.