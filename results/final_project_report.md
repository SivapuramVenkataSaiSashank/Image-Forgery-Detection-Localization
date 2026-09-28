# Final Year Project: Comparative Analysis & Results

## 1. Executive Summary
This project investigates and benchmarks **Convolutional Neural Networks (ResNet-50)** vs **Vision Transformers (Swin-T)** for Image Manipulation Detection and Localization on the DEFACTO & COCO 2017 datasets, and proposes a **Hybrid CNN+ViT Dual-Task Architecture** with multi-scale feature fusion and a U-Net style segmentation decoder.

---

## 2. Quantitative Performance Table

| Model / Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score | ROC-AUC | Localization (Mean IoU) | Localization (Mean Dice) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** *(CNN Baseline - EXP 2)* | 81.97% | 83.90% | 79.13% | 0.8145 | 0.9078 | *Coarse Heatmap* | — |
| **Swin-T** *(ViT Baseline - EXP 3)* | 84.38% | 85.46% | 82.87% | 0.8414 | 0.9317 | *Attention Rollout* | — |
| **Proposed Hybrid Model** *(EXP 9)* | **86.80%** | **87.62%** | **85.70%** | **0.8665** | **0.9518** | **57.05%** | **62.32%** |
| **Improvement ($\\Delta$)** | **+4.83%** *(vs CNN)* | **+3.72%** *(vs CNN)* | **+6.57%** *(vs CNN)* | **+0.052** | **+0.044** | **Pixel-level precision** | **High boundary fidelity** |

---

## 3. Saved Artifacts in `results/`
1. `comparative_metrics_table.csv`: Raw metrics table.
2. `comparative_roc_curves.png`: High-resolution comparative ROC curves.
3. `comparative_confusion_matrices.png`: Side-by-side confusion matrix comparisons.
4. `visual_comparison_benchmark.png`: 5-column qualitative localization visual comparison.
