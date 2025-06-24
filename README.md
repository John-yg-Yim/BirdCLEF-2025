# 🐦 BirdCLEF+2025 - Bird Sound Classification (Kaggle Competition)

This repository contains the final code used in the BirdCLEF+2025 competition,
where the objective was to classify 206 species — primarily birds, but also includes amphibians, mammals, and insects — from 60-second soundscape recordings.

Our best model was trained from scratch using EfficientNetV2 and custom preprocessing,
and an ensemble version using public NFNet + SE-ResNeXt models achieved Bronze medal-level performance.

---

📁 **Key Files**

| Filename | Description |
|----------|-------------|
| `Preprocessing.ipynb` | Audio preprocessing pipeline (mel-spectrogram, segment selection, etc.) |
| `Train.ipynb` | Custom model training notebook (EfficientNetV2, focal loss, CV) |
| `Inference.ipynb` | Inference code for the above model |
| `Ensemble.ipynb` | Weighted blend using NFNet + SE-ResNeXt (public notebooks) |
| `BirdCLEF.ipynb` | Full competition summary, leaderboard progression, analysis (used for portfolio) |
| `BirdCLEF.html` | HTML version of the above |

---

🧪 **Result Summary**

| Model | Public LB | Private LB | Submitted |
|-------|-----------|------------|-----------|
| Custom Model | 0.835 | 0.847 | O |
| Public Ensemble | 0.878 | 0.893 | X |
| Public Ensemble | 0.881 | 0.891 | O |
| Final Rank | **162 / 2026** | **507 / 2026** | - |

- Model backbone: `tf_efficientnetv2_s.in21k_ft_in1k` via `timm`
- Audio features: Mel-spectrograms (512 bins), frequency-based segment filtering
- Ensemble: Weighted blend (NFNet 25%, SE-ResNeXt 75%) with post-processing

---

🔍 **Dataset Overview**

- `train_audio`: ~20K weakly labeled bird audio clips (32kHz OGG)
- `train_soundscapes`: ~10K unlabeled 60s recordings
- `test_soundscapes`: ~700 evaluation files (5s segments)

---

📝 **Note**

This repository is for code backup purposes.  
Model weights and datasets are not included.
