# 🌾 Crop Stress Detection using NDVI & Deep Learning (Punjab, India)

This project builds an end-to-end **Remote Sensing + Deep Learning pipeline** to classify agricultural crop health/stress levels using **Sentinel-2 satellite imagery (GeoTIFF)** from Punjab, India.

We extract NDVI-based patch datasets and train multiple ML/DL models including **Custom CNN**, **ResNet Transfer Learning**, and **Temporal CNN + LSTM** for vegetation stress classification.

---

## 🎯 Project Goals

✅ Generate NDVI maps from satellite imagery  
✅ Convert NDVI into structured patch datasets  
✅ Classify crop stress into **3 vegetation classes**  
✅ Compare baseline vs deep learning vs temporal models  
✅ Track experiments using **MLflow**

---

## 🛰 Dataset Description

Satellite imagery exported from **Google Earth Engine (GEE)**.

### 📌 Source
- Sentinel-2 Surface Reflectance (S2 SR)

### 📌 Study Region
- Punjab, India

### 📌 Seasons Used
- **Rabi 2023**
- **Kharif 2023**

### 📌 Temporal Data (Phase 3)
NDVI stack for:
- January 2023
- February 2023
- March 2023

---

## 🌿 NDVI Computation

NDVI is computed using:

\[
NDVI = \frac{NIR - RED}{NIR + RED}
\]

Where:
- **Red = Band 4**
- **NIR = Band 8**

---

## 🧩 Patch Creation

The GeoTIFF is split into **32×32 patches**.

Each patch is labeled based on mean NDVI:

| NDVI Range | Class | Meaning |
|-----------|-------|---------|
| < 0.3     | 0     | Low vegetation / stressed |
| 0.3–0.6   | 1     | Medium vegetation |
| > 0.6     | 2     | High vegetation / healthy |

---

## 🧠 Models Implemented

---

### ✅ Model 1: NDVI Threshold Baseline (Rule-Based)

A simple scientific heuristic:

- NDVI < 0.3 → Stressed  
- 0.3 ≤ NDVI < 0.6 → Moderate  
- NDVI ≥ 0.6 → Healthy  

⭐ **Explainability: 10/10**  
⭐ **Interpretability: 10/10**  
❌ Cannot learn spatial patterns

---

### ✅ Model 2: Custom CNN (Single-Date Deep Learning)

A CNN trained from scratch on NDVI patches.

**Learns spatial patterns like:**
- crop texture
- irrigation patterns
- field-level consistency

⭐ **Explainability: 6/10**  
⭐ **Accuracy: ~95–98%**

---

### ✅ Model 3: ResNet18 Transfer Learning (Single-Date Multiband)

A pretrained **ResNet18** (ImageNet) fine-tuned on satellite patch dataset.

**Inputs can include:**
- NDVI only
- Red + NIR + NDVI
- RGB (B4, B3, B2)
- Multiband Sentinel-2

⭐ **Accuracy: ~98–99% (Best single-date performance)**  
⭐ **Generalization: Very strong across Rabi + Kharif**

---

### ✅ Model 4: Temporal CNN (NDVI Stack)

Uses NDVI of multiple months as multi-channel input.

Input shape:

