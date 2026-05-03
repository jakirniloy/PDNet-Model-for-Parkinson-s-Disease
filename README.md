# 🧠 PDNet: Deep Learning Framework for Parkinson's Disease Detection

> 🚀 A Multi-Modal AI Framework for Early Parkinson's Disease Diagnosis using MRI, PET, and DaTscan

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](#license)
[![Status](https://img.shields.io/badge/Status-Research-purple.svg)](#)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#️-architecture)
- [Dataset](#-dataset)
- [Methodology](#️-methodology)
- [Results](#-results)
- [Time Complexity](#️-time-complexity)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Explainability](#-explainability)
- [Authors](#️-authors)
- [Future Work](#-future-work)
- [License](#-license)

---

## 📖 Overview

Parkinson's Disease (PD) is frequently **misdiagnosed in early stages (~50% error rate)**, making timely detection extremely challenging. **PDNet** is a powerful deep learning framework designed to improve diagnostic accuracy by combining complementary neuroimaging modalities.

PDNet integrates:

| Modality | Description |
|----------|-------------|
| 🧠 **MRI** | Structural brain anatomy |
| 🔬 **PET** | Metabolic brain activity |
| 📊 **DaTscan** | Dopamine transporter imaging |

Together, these provide a **robust, interpretable, and efficient** diagnostic pipeline.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔗 **Multi-Modal Learning** | Fuses MRI, PET, and DaTscan for richer representation |
| 🧠 **Attention-Based Fusion** | Learns to weight each modality dynamically |
| ⚙️ **Learnable Ensemble** | Stacked generalization for optimal model combination |
| 🔍 **Grad-CAM Explainability** | Visual heatmaps over clinically relevant brain regions |
| 🎯 **Uncertainty Estimation** | Monte Carlo Dropout for prediction confidence |
| 🚫 **No Data Leakage** | Strict subject-level train/test splitting |
| 📊 **5-Fold Cross Validation** | Statistically robust performance evaluation |

---

## 🏗️ Architecture

PDNet is a modular multi-model framework combining transfer learning backbones with custom lightweight architectures.

```
Input (MRI + PET + DaTscan)
        │
        ▼
┌───────────────────────────────────────┐
│         Feature Extraction Layer       │
│                                       │
│  ResNet50 │ EfficientNet-B0 │ InceptionV3 │
│  DenseNet121 │ InceptionResNetV2     │
│  Custom CNN │ LoCNN (Lightweight)   │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│     Attention-Based Feature Fusion     │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│      Learnable Weighted Ensemble       │
│       (Stacked Generalization)         │
└───────────────────────────────────────┘
        │
        ▼
   Diagnosis Output
   (PD / Healthy)
```

### 🔹 Transfer Learning Backbones

- **ResNet50** — Deep residual learning with skip connections
- **EfficientNet-B0** — Compound scaling for efficiency
- **InceptionV3** — Multi-scale feature extraction
- **DenseNet121** — Dense connectivity pattern
- **InceptionResNetV2** — Combined Inception + ResNet architecture

### 🔹 Custom Models

- **Custom CNN** — Task-specific convolutional network
- **LoCNN** — Lightweight CNN optimized for speed

### 🔹 Fusion & Ensemble

- **Attention-Based Feature Fusion** — Adaptive cross-modal weighting
- **Learnable Weighted Ensemble** — Meta-learner over base model predictions

---

## 📂 Dataset

| Source | Modality | Participants |
|--------|----------|-------------|
| PPMI Dataset | MRI, PET, DaTscan | Primary source |
| NTUA Dataset | MRI | Supplementary |

**Total Summary:**

```
Total Participants : 1,195
├── DaTscan        :   642
├── MRI            :   402
└── PET            :   151
```

> ⚠️ Data is not included in this repository. Access PPMI data at [ppmi-info.org](https://www.ppmi-info.org) after registration.

---

## ⚙️ Methodology

```
1. Subject-Level Data Splitting   →  Prevents data leakage across folds
2. 5-Fold Cross-Validation        →  Robust generalization estimate
3. Feature Extraction             →  Pretrained + custom CNN backbones
4. Attention-Based Fusion         →  Weighted cross-modal integration
5. Learnable Ensemble             →  Stacked meta-learner
6. Evaluation                     →  Accuracy, AUC, F1, Sensitivity, Specificity
```

### Subject-Level Splitting (Critical)

All images from the same patient are kept **exclusively in either train or test**, never split across both. This prevents inflated performance metrics caused by data leakage — a common pitfall in medical imaging research.

---

## 📈 Results

| Metric | Performance |
|--------|-------------|
| 🎯 Accuracy | High across all modalities |
| 📉 Balanced Performance | Strong sensitivity & specificity |
| ⚡ Efficiency | Optimal time-performance tradeoff |
| 🔍 Interpretability | Clinically meaningful Grad-CAM maps |

> Full quantitative results are available in the associated research paper and experiment notebooks.

---

## ⏱️ Time Complexity

| Model Category | Operations | Characteristics |
|----------------|------------|-----------------|
| Heavy (DenseNet121, IRv2) | 10¹¹ – 10¹² | High accuracy, slow training |
| Medium (ResNet50, InceptionV3) | 10⁹ – 10¹⁰ | Good balance |
| Lightweight (LoCNN, Custom CNN) | 10⁷ – 10⁸ | Fast, mobile-friendly |
| **PDNet Ensemble** | **~20K params** | **Optimal balance ✅** |

PDNet achieves high predictive performance with moderate parameter count and efficient execution, making it suitable for real-world clinical deployment.

---

## 📁 Repository Structure

```
PDNet-Model-for-Parkinson-s-Disease/
│
├── 📓 Notebooks/
│   ├── VGG16_PD.ipynb
│   ├── ResNet50_PD.ipynb
│   ├── EfficientNetB0_PD.ipynb
│   ├── InceptionV3_PD.ipynb
│   ├── InceptionResNetV2_PD.ipynb
│   ├── ViT_PD.ipynb
│   ├── CustomCNN_PD.ipynb
│   ├── LoCNN_PD.ipynb
│   ├── DataAugmentation.ipynb
│   └── KFold_Experiments.ipynb
│
├── 🐍 helper_functions.py       # Shared utilities and preprocessing
├── 📄 README.md                 # Project documentation
└── 📋 requirements.txt          # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/jakirniloy/PDNet-Model-for-Parkinson-s-Disease.git

# 2. Navigate into the project
cd PDNet-Model-for-Parkinson-s-Disease

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

### Requirements

```txt
tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
jupyter>=1.0.0
```

### Running a Notebook

1. Open any notebook from the `Notebooks/` directory
2. Set the dataset path variable at the top of the notebook
3. Run all cells sequentially (`Kernel → Restart & Run All`)

---

## 🔍 Explainability

PDNet uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to produce interpretable visual explanations.

### Highlighted Brain Regions

```
✅ Substantia Nigra   — dopaminergic neuron degeneration site
✅ Putamen            — motor control region affected in PD
✅ Basal Ganglia      — primary area of dopamine loss
```

Grad-CAM ensures the model attends to **clinically relevant anatomical structures**, building trust for medical practitioners and enabling human-in-the-loop verification.

---

## 👨‍💻 Authors

**Department of Computer Science & Engineering**  
**East West University, Dhaka, Bangladesh**\\
**Department of Computer Science and Engineering** 
**Brac University, Dhaka, Bangladesh**

| Name | Role |
|------|------|
| Md. Shakil Bhuiyan | Core Contributor |
| Yusuf Salehin | Core Contributor |
| Md Jakir Hossain | Core Contributor |
| Mariya Rahman Momo | Contributor |
| Md Towfiq Bin Towhid | Contributor |
| Md. Toufiq Sarrowar | Contributor |
| Ryan Mohammad Bin Shahjahan | Contributor |
| Md Jubaeid Ali | Contributor |
| Tasnim Israk Synthia | Contributor |
| Ahmed Wasif Reza | Supervisor / Contributor |

---

## 💡 Future Work

- [ ] 🏥 **Clinical Deployment** — Integration with hospital diagnostic pipelines
- [ ] 📱 **Real-Time Diagnostic Tools** — Mobile and edge-device inference
- [ ] 🧠 **Additional Modalities** — fMRI, EEG, speech and gait data
- [ ] 🤖 **Improved Lightweight Architectures** — Further compression for resource-constrained environments
- [ ] 🌐 **Federated Learning** — Privacy-preserving training across hospitals

---

## 📜 License

This project is intended for **research and academic purposes only**. Commercial use is not permitted without explicit written permission from the authors.

---

## ⭐ Support

If you find PDNet useful in your research, please consider:

- ⭐ **Starring** this repository
- 🍴 **Forking** it for your own experiments
- 📢 **Sharing** it with your research community
- 📝 **Citing** our work if used in a publication

---

<div align="center">

🚀 **Advancing AI for Healthcare — Early Detection Saves Lives**

*PDNet | East West University | Dhaka, Bangladesh*

</div>
