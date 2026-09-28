# Walkthrough: VeriSight AI Forensics Web Application

We have created the full-stack web application located in the [`web/`](file:///f:/%23sem7/22AIE498%20Project%20Phase%201/Project/CNN-vs-ViT-for-Image-Manipulation-Detection-main/web) directory for real-time deepfake & image manipulation detection and pixel-level boundary localization.

---

## 🌟 Application Features

1. **Dual-Task AI Inference Engine**:
   * Powered by the **Proposed Hybrid CNN+ViT Dual-Task model** (`exp9/hybrid_best.pth`) with an 86.80% test accuracy and 0.9518 ROC-AUC.
   * Model selector allowing instant comparison with baseline **ResNet-50 (EXP 2)** and **Swin-T (EXP 3)** models.
2. **Real vs. Fake Classification & Confidence Scoring**:
   * Real-time probability estimation with animated radial circular gauges and glowing verdict status cards.
3. **Interactive Manipulation Localization Studio**:
   * **Split-Slider View**: Drag to compare original photos vs high-resolution tampering localization heatmaps.
   * **Colormap Customization**: Choose between *Jet (Rainbow)*, *Thermal (Hot)*, *Turbo Neon*, and *Viridis*.
   * **Binary Mask & Overlay Toggle**: Switch between transparent heatmap overlays and raw binary segmentation masks.
4. **Forensics Telemetry Breakdown**:
   * Calculated **Tampered Region Area (%)**.
   * **Peak Anomaly Intensity (%)**.
   * Real-time inference latency (milliseconds) and hardware accelerator status.
5. **1-Click Test Gallery**:
   * Curated quick-test samples across Authentic photos and DEFACTO manipulation types (Splicing, Copy-Move, Inpainting, and Face Swapping).

---

## 📷 Screenshots & Visual Demo

| Authentic Photo Analysis | Manipulated Photo Analysis |
| :---: | :---: |
| ![Authentic Banner](file:///C:/Users/skous/.gemini/antigravity-ide/brain/a340c04c-214c-4720-9696-3b9ff89db9cf/authentic_banner_clean_1788236225537.png) | ![Fake Verdict Banner](file:///C:/Users/skous/.gemini/antigravity-ide/brain/a340c04c-214c-4720-9696-3b9ff89db9cf/fake_verdict_banner_1788235924757.png) |

| Interactive Split-Slider Studio |
| :---: |
| ![Split-Slider Studio](file:///C:/Users/skous/.gemini/antigravity-ide/brain/a340c04c-214c-4720-9696-3b9ff89db9cf/studio_split_slider_70_1788236061089.png) |

---

## 🚀 How to Access and Run the Application

The Flask server is currently **active and running**:

* **Local URL**: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

To launch it manually in any terminal at any time:
```powershell
& "F:\#sem7\22AIE498 Project Phase 1\Project\gpu_env\Scripts\python.exe" web/app.py
```
