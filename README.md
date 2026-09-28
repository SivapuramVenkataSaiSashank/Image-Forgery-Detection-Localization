# Image Manipulation Detection: CNN vs. ViT & Proposed Hybrid Architecture

This repository contains the complete experimental suite, trained models, visual interpretability benchmarks, and a full-stack forensic web application for **Image Manipulation Detection and Localization**, comparing Convolutional Neural Networks (ResNet-50) and Vision Transformers (Swin Transformer), alongside our **Proposed Hybrid CNN + ViT Dual-Task Solution**.

---

## 🔬 Experiments Overview

### Experiment 1: Data Preparation & Dataset Statistics
**Notebook:** `Final Year.ipynb` (Cells 1–2)  
**Results Folder:** `exp1/`  
**Description:**  
Preprocessing of the multi-source dataset (COCO 2017 & DEFACTO across Splicing, Copy-Move, Inpainting, and Face Manipulation), data cleaning, label encoding, stratified train/validation/test split generation, and dataset statistics (`train_data.pkl`, `val_data.pkl`, `test_data.pkl`, `data_stats.csv`).

### Experiment 2: CNN Baseline Classification (ResNet-50)
**Notebook:** `Final Year.ipynb` (Cells 3–4)  
**Results Folder:** `exp2/`  
**Description:**  
Implements ResNet-50 as the deep convolutional baseline for image manipulation detection. Includes training, validation, evaluation on the test set, and metrics calculation (81.97% Accuracy, 0.9078 ROC-AUC, 81.97% F1-Score).

### Experiment 3: ViT Baseline Classification (Swin Transformer)
**Notebook:** `Final Year.ipynb` (Cells 5–6)  
**Results Folder:** `exp3/`  
**Description:**  
Implements Swin Transformer (`swin_tiny_patch4_window7_224`) as the hierarchical Vision Transformer baseline. Evaluates shifted-window self-attention for manipulation detection (84.38% Accuracy, 0.9317 ROC-AUC, 84.38% F1-Score).

### Experiment 4: Confusion Matrix & Comparative Error Analysis
**Notebook:** `Final Year.ipynb` (Cells 7–8)  
**Results Folder:** `exp4/`  
**Description:**  
Generates side-by-side normalized confusion matrices and cross-architecture error analyses for CNN and ViT models, evaluating True Positive, False Positive, True Negative, and False Negative distribution patterns.

### Experiment 5: Qualitative Explainability Visualization (Grad-CAM vs. Attention Rollout)
**Notebook:** `Final Year.ipynb` (Cells 9–10)  
**Results Folder:** `exp5/`  
**Description:**  
Visualizes and compares model interpretability using Grad-CAM for ResNet-50 and Attention Rollout for Swin-T across manipulation types. Highlights the classification-localization gap between CNN edge sensitivity and ViT global context.

### Experiment 6 & 7: Quantitative Explainability Evaluation & Pointing Game
**Notebook:** `Final Year.ipynb` (Cells 11–12)  
**Results Folder:** `exp6_7/`  
**Description:**  
Quantitatively evaluates explainability maps against ground-truth manipulation masks using Intersection over Union (IoU), Energy Inside Mask (EIM), and Pointing Game Accuracy (Hit/Miss localization rates).

### Experiment 8: Failure Case Analysis & Boundary Discrepancies
**Notebook:** `Final Year.ipynb` (Cells 13–14)  
**Results Folder:** `exp8/`  
**Description:**  
In-depth diagnostic analysis of failure cases where CNN and ViT models misclassify or fail to localize manipulations, examining subtle boundary blending, low-contrast inpainting, and fine-grained splicing artifacts.














































### Experiment 9: Proposed Hybrid CNN + Vision Transformer Dual-Task Solution
**Notebook:** `Final Year.ipynb` (Cells 15–17)  
**Results Folder:** `exp9/`, `results/`  
**Description:**  
Proposes a novel **Hybrid CNN + Vision Transformer** dual-task architecture that bridges the classification-localization gap:
- **Multi-Scale Feature Fusion:** Fuses hierarchical multi-scale feature maps from ResNet-50 (local texture details) and Swin Transformer (global semantic context) across 4 spatial stages ($56\times56$, $28\times28$, $14\times14$, $7\times7$).
- **U-Net Localization Decoder:** Combines a global classification head with a U-Net/FPN-style segmentation decoder with lateral skip-connections for dense pixel-level manipulation boundary localization.
- **Composite Loss & Curriculum Learning:** Optimized using a combined loss function ($\mathcal{L} = \alpha \mathcal{L}_{\text{BCE}} + \beta \mathcal{L}_{\text{Dice}} + \gamma \mathcal{L}_{\text{IoU}} + \delta \mathcal{L}_{\text{Boundary}} + \lambda_{\text{aux}} \mathcal{L}_{\text{Aux}}$) with attention-guided refinement and dynamic 3-stage curriculum training.
- **Benchmark Performance:** Outperforms individual baselines, achieving **86.80% Classification Accuracy**, **0.9518 ROC-AUC**, **57.05% Mean IoU**, and **62.32% Mean Dice**.

---

## 🌐 Web Application (`web/`)

A full-stack, interactive **Forensics Web Studio** built with Flask, Vanilla CSS, and modern JavaScript:
- **Joint Decision Fusion:** Combines classification logits and U-Net mask density to eliminate false alarms on real-world photos.
- **Aspect-Ratio Preserving Preprocessing:** Prevents geometric distortion on arbitrary camera/smartphone images.
- **Interactive Split-Slider Studio:** Real-time visual comparison between original images and tampering heatmaps.
- **Custom Colormaps & Mask Toggles:** Jet, Thermal Hot, Turbo Neon, and Viridis colormaps with pure binary mask overlays.
- **Forensic Telemetry:** Computes localized tampering area percentage, peak anomaly intensity, and inference latency in milliseconds.

To start the web application:
```powershell
& "gpu_env\Scripts\python.exe" web/app.py
```
Open **`http://127.0.0.1:5000/`** in your browser.

---

## 📊 Consolidated Results (`results/`)

- **`comparative_metrics_table.csv`**: Quantitative metric benchmark across all three models.
- **`comparative_roc_curves.png`**: Comparative ROC curve plot with AUC scores.
- **`comparative_confusion_matrices.png`**: 3-panel normalized confusion matrix comparison.
- **`visual_comparison_benchmark.png`**: 5-column qualitative benchmark (Original, Ground Truth, Grad-CAM, Swin-T Attention, Proposed Hybrid).

---

## ⚙️ Notes & Environment Setup

- **Unified Notebook:** All 9 experiments are implemented sequentially in `Final Year.ipynb`.
- **Environment:** Configured in `gpu_env` with CUDA GPU support.
- **Dependencies:** `torch`, `torchvision`, `timm`, `matplotlib`, `seaborn`, `numpy`, `pandas`, `pillow`, `flask`, `flask-cors`.
- **Reproducibility:** All models and experiment evaluation pipelines include exact random seeds and serialized checkpoints in their respective `exp*/` folders.
