# Bounded WBCE on CheXpert

Code accompanying **Weight Rescaling in Weighted Cross-Entropy: Balancing Discrimination and Calibration in Long-Tailed Chest Radiographs**, Gustavo Pedreira, Henrique Fernandes, and Marcelo Zanchetta, WVC 2026.

The method rescales prevalence-based positive weights to `[1, lambda_max]` in weighted binary cross-entropy. The backbone is DenseNet-121 with 14 outputs. The paper compares weight limits 1, 3, and 5 across seeds 2026, 123, and 456, using U-Soft targets of 0.55.

## Setup

Use Python 3.11 or newer. Training requires a CUDA GPU; FP32 evaluation also works on CPU.

```bash
pip install -r requirements.txt
```

Obtain the CheXpert small dataset through the [official project](https://stanfordmlgroup.github.io/competitions/chexpert/). Place `train.csv`, `valid.csv`, and their `train/` and `valid/` image folders under the same dataset directory. Images and trained weights are not included.

## Run

Train the paper's nine-model comparison:

```bash
python wvc.py train --data-dir /path/to/chexpert --output-root runs/wvc
```

To repeat the full sequence (uncertainty-policy comparison, weight screening, and the nine-model comparison), add `--phase all`. Phase 1 ranks five policies on development data. Phases 2a/2b default to the paper's fixed U-Soft policy; `--strategy` selects an alternative explicitly. A saved patient split is reused across phases. Completed matching baseline runs are reused. Use `--resume` to continue an interrupted experiment.

Evaluate the models produced by your training:

```bash
python wvc.py evaluate --data-dir /path/to/chexpert --experiment-root runs/wvc
```

The command prints a new evaluation directory containing model-level metrics, mean/sample SD, per-class metrics, and the paired image bootstrap for seed 2026. Generate the figure from that directory:

```bash
python wvc.py figures --evaluation-dir runs/evaluation/PRINTED_DIRECTORY
```

For Google Colab, open [colab.ipynb](colab.ipynb), choose a GPU, and set the dataset and output paths before running training.

## Protocol

Images are resized to 320 x 320 and normalized with ImageNet statistics. Training adds random rotations of up to 10 degrees with probability 0.5. This preprocessing applies consistently across training, development, and evaluation.

The development set contains approximately 10% of training patients, stratified by positive Pneumothorax/Pneumonia labels with seed 2026. AdamW uses learning rate 1e-4, weight decay 1e-2, batch size 128, and at most 12 epochs. Checkpoints maximize development Focus-6 Macro-F1. Class thresholds maximize development F1 and remain frozen during official evaluation.

Phase 1/2a development ECE uses zero-filled targets without the binary mask, matching the reported screening procedure. Official ECE pools the two tail classes with up to ten quantile bins. Weight-limit comparisons are exploratory. Historical tail Macro-F1 increased from 0.1691 to 0.2004 while ECE increased from 0.0421 to 0.0911; fresh training results can vary. This cleaned training implementation has not been rerun end to end on CheXpert. Resuming does not restore worker/augmentation RNG state exactly.

## Citation

If you use this code, please cite the accompanying paper:

G. Pedreira, H. Fernandes, and M. Zanchetta, “Weight Rescaling in Weighted Cross-Entropy: Balancing Discrimination and Calibration in Long-Tailed Chest Radiographs,” Workshop on Computer Vision (WVC), 2026, accepted.

Released under the MIT license.
