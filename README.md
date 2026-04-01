---
title: DK DermInsights
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.55.0"
app_file: app.py
pinned: true
license: mit
---

# DK DermInsights — Skin Lesion Classification

AI-powered skin lesion classification tool combining dermatoscopic image analysis with clinical risk factors.

**Author:** Daria Khotuleva — Dermatologist & Data Scientist

- **Model:** Multimodal EfficientNet-B0 (image + patient metadata)
- **Dataset:** ISIC 2019 + HAM10000 + ISIC 2020 — 31,331 dermatoscopic images
- **Classes:** 8 types of skin lesions (including melanoma, BCC, SCC)
- **Features:** Grad-CAM visualization, clinical questionnaire, threshold tuning for melanoma
