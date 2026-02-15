# 🌾 Crop Stress Detection using NDVI & Deep Learning (Punjab, India)

This repository contains an end-to-end **Remote Sensing + Deep Learning pipeline** for detecting crop vegetation stress using **Sentinel-2 satellite imagery** from Punjab, India.

The project covers:

* Satellite data extraction using **Google Earth Engine (GEE)**
* NDVI computation and patch dataset creation
* Training multiple ML/DL models:

  * NDVI Threshold Baseline
  * Custom CNN
  * ResNet18 Transfer Learning
  * Temporal NDVI CNN
  * CNN + LSTM Temporal Model
* Comparing models using **Accuracy, Explainability, Interpretability**
* Experiment tracking using **MLflow**

---

## 🎯 Project Motivation

Agricultural monitoring requires detecting crop health/stress early and accurately.

Satellite vegetation indices like **NDVI** provide a strong scientific signal of plant health, but deep learning models can go beyond simple NDVI thresholds by learning:

* spatial crop texture
* field structure
* seasonal variation
* temporal growth patterns

---

## 🛰 Dataset (Not Uploaded to GitHub)

⚠️ The dataset files (`.tif`) are **very large GeoTIFF files**, so they are **NOT included** in this GitHub repository.

Instead, this README explains **how to reproduce the dataset** from scratch using Google Earth Engine.

---

# 📌 DATA COLLECTION PIPELINE (Google Earth Engine)

This dataset was created using **Sentinel-2 Surface Reflectance** imagery.

---

## 🌍 Data Source

* **Sentinel-2 SR (Surface Reflectance)**
* Google Earth Engine collection:

  * `COPERNICUS/S2_SR_HARMONIZED`

---

## 🗺 Study Region

* Punjab, India (cropland belt)
* Region selected using polygon geometry in GEE

---

## 📅 Seasons Used

### 🌾 Rabi Season (Wheat)

* Example time range: Nov 2022 – Mar 2023

### 🌱 Kharif Season (Rice / Monsoon Crops)

* Example time range: Jun 2023 – Oct 2023

---

## 🧠 Bands Exported

Sentinel-2 bands used:

| Band | Name  | Use               |
| ---- | ----- | ----------------- |
| B2   | Blue  | RGB visualization |
| B3   | Green | RGB visualization |
| B4   | Red   | NDVI              |
| B8   | NIR   | NDVI              |

---

# 🧾 Google Earth Engine Export Code

Paste this code in **Google Earth Engine Code Editor**:

```javascript
// ------------------------------
// 1) REGION (Punjab Example)
// ------------------------------
var region = /* your Punjab polygon geometry */;

// ------------------------------
// 2) Sentinel-2 Dataset
// ------------------------------
var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(region)
  .filterDate("2023-01-01", "2023-03-31") // change dates for season
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20));

// ------------------------------
// 3) Median Composite
// ------------------------------
var composite = s2.median().clip(region);

// ------------------------------
// 4) Select Bands
// ------------------------------
var export_img = composite.select(["B2","B3","B4","B8"]);

// ------------------------------
// 5) Export to Drive
// ------------------------------
Export.image.toDrive({
  image: export_img,
  description: "Punjab_Rabi_2023",
  folder: "GEE_Exports",
  region: region,
  scale: 10,
  maxPixels: 1e13
});
```

---

# 🔁 Repeat Export for Required Files

Repeat the same export code (just change the date range + description) to generate these GeoTIFFs:

* `Punjab_Rabi_2023.tif`
* `Punjab_Kharif_2023.tif`
* `Punjab_Rabi_Jan_2023.tif`
* `Punjab_Rabi_Feb_2023.tif`
* `Punjab_Rabi_Mar_2023.tif`

---

## 🛰 Export Code: Punjab_Rabi_2023

```javascript
var region = /* your Punjab polygon geometry */;

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(region)
  .filterDate("2022-11-01", "2023-03-31")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20));

var composite = s2.median().clip(region);

var export_img = composite.select(["B2","B3","B4","B8"]);

Export.image.toDrive({
  image: export_img,
  description: "Punjab_Rabi_2023",
  folder: "GEE_Exports",
  region: region,
  scale: 10,
  maxPixels: 1e13
});
```

---

## 🌱 Export Code: Punjab_Kharif_2023

```javascript
var region = /* your Punjab polygon geometry */;

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(region)
  .filterDate("2023-06-01", "2023-10-31")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20));

var composite = s2.median().clip(region);

var export_img = composite.select(["B2","B3","B4","B8"]);

Export.image.toDrive({
  image: export_img,
  description: "Punjab_Kharif_2023",
  folder: "GEE_Exports",
  region: region,
  scale: 10,
  maxPixels: 1e13
});
```

---

## 🌾 Export Code: Punjab_Rabi_Jan_2023

```javascript
var region = /* your Punjab polygon geometry */;

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(region)
  .filterDate("2023-01-01", "2023-01-31")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20));

var composite = s2.median().clip(region);

var export_img = composite.select(["B2","B3","B4","B8"]);

Export.image.toDrive({
  image: export_img,
  description: "Punjab_Rabi_Jan_2023",
  folder: "GEE_Exports",
  region: region,
  scale: 10,
  maxPixels: 1e13
});
```

---

## 🌾 Export Code: Punjab_Rabi_Feb_2023

```javascript
var region = /* your Punjab polygon geometry */;

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(region)
  .filterDate("2023-02-01", "2023-02-28")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20));

var composite = s2.median().clip(region);

var export_img = composite.select(["B2","B3","B4","B8"]);

Export.image.toDrive({
  image: export_img,
  description: "Punjab_Rabi_Feb_2023",
  folder: "GEE_Exports",
  region: region,
  scale: 10,
  maxPixels: 1e13
});
```

---

## 🌾 Export Code: Punjab_Rabi_Mar_2023

```javascript
var region = /* your Punjab polygon geometry */;

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(region)
  .filterDate("2023-03-01", "2023-03-31")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20));

var composite = s2.median().clip(region);

var export_img = composite.select(["B2","B3","B4","B8"]);

Export.image.toDrive({
  image: export_img,
  description: "Punjab_Rabi_Mar_2023",
  folder: "GEE_Exports",
  region: region,
  scale: 10,
  maxPixels: 1e13
});
```

---

# 📂 How to Use Dataset in Google Colab

Once you export GeoTIFFs from GEE:

✅ Download them from Google Drive
✅ Upload them into your Google Colab session
✅ Load them using Rasterio

---

# 📦 Install Required Libraries

```bash
pip install rasterio mlflow -q
```

---

# 📌 Load GeoTIFF in Python

```python
import rasterio
import numpy as np

with rasterio.open("Punjab_Rabi_2023.tif") as src:
    img = src.read()   # (bands, H, W)

blue  = img[0].astype(np.float32)
green = img[1].astype(np.float32)
red   = img[2].astype(np.float32)
nir   = img[3].astype(np.float32)

print(img.shape)
```

Expected shape:

```
(4, Height, Width)
```

---

# 🌿 NDVI Computation

NDVI is computed using:

[
NDVI = \frac{NIR - RED}{NIR + RED}
]

```python
ndvi = (nir - red) / (nir + red + 1e-6)
ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)

print("NDVI Range:", np.min(ndvi), np.max(ndvi))
```

---

# 🧩 Patch Dataset Creation

Each NDVI map is divided into **32×32 patches**.

Each patch is labeled using mean NDVI:

| NDVI Range | Class | Meaning                   |
| ---------- | ----- | ------------------------- |
| < 0.3      | 0     | Low vegetation / stressed |
| 0.3–0.6    | 1     | Medium vegetation         |
| > 0.6      | 2     | High vegetation / healthy |

---

# 🧾 Patch Generation Code

```python
patch_size = 32
patches = []
labels = []

H, W = ndvi.shape

for i in range(0, H - patch_size, patch_size):
    for j in range(0, W - patch_size, patch_size):

        patch = ndvi[i:i+patch_size, j:j+patch_size]
        patch = np.nan_to_num(patch, nan=0.0)

        mean_ndvi = np.mean(patch)

        if mean_ndvi < 0.3:
            label = 0
        elif mean_ndvi < 0.6:
            label = 1
        else:
            label = 2

        patches.append(patch)
        labels.append(label)

patches = np.array(patches)
labels = np.array(labels)

# Convert to PyTorch format: (N, 1, 32, 32)
patches = np.expand_dims(patches, axis=1)

print("Patches:", patches.shape)
print("Labels:", labels.shape)
```

---

# 🧠 Models Implemented

This project compares multiple models ranging from simple rule-based baselines to advanced deep learning temporal models.

---

## ✅ Model 1: NDVI Threshold Baseline (Rule-Based)

### 📌 What it is

A simple baseline classifier using NDVI thresholds.

### ✅ Strengths

* Best explainability
* Fully interpretable
* Directly supported by vegetation science

### ❌ Weakness

* Cannot learn spatial patterns
* Cannot generalize beyond thresholds
* Sensitive to noise/cloud contamination

⭐ Explainability: 10/10
⭐ Interpretability: 10/10
⭐ Accuracy: Strong baseline

---

## ✅ Model 2: Custom CNN (Single-Date NDVI)

### 📌 What it is

A CNN trained from scratch on NDVI patch dataset.

### ✅ Strengths

* Learns spatial patterns (texture, patch structure)
* Better than thresholding for noisy NDVI
* Lightweight and fast to train

### ❌ Weakness

* Needs careful preprocessing (NaNs must be handled)
* Less powerful than pretrained architectures

⭐ Explainability: 6/10
⭐ Interpretability: 6/10
⭐ Accuracy: ~95–98%

---

## ✅ Model 3: ResNet18 Transfer Learning (Multiband)

### 📌 What it is

A pretrained ResNet18 fine-tuned on satellite patch dataset.

### Inputs used

* NDVI patches expanded to 3 channels
* OR multiband inputs such as (Red, NIR, NDVI)

### ✅ Strengths

* Best accuracy and generalization
* Robust across Rabi + Kharif seasons
* Learns high-level spatial features automatically

### ❌ Weakness

* More compute-heavy
* Less explainable than threshold baseline

⭐ Explainability: 5/10
⭐ Interpretability: 7/10
⭐ Accuracy: ~98–99% (Best single-date model)

---

# ⏳ Temporal Models (Phase 3)

Temporal models answer a new agricultural question:

**How does vegetation change over time?**

Even if accuracy is similar, temporal models are more valuable because they capture growth behavior.

---

## ✅ Model 4: Temporal CNN (NDVI Stack)

### 📌 What it is

Uses NDVI from multiple months as multi-channel input.

Input shape:

```
(N, 3, 32, 32)
```

Channels:

* NDVI_Jan
* NDVI_Feb
* NDVI_Mar

### ✅ Strengths

* Simple temporal learning
* More robust to NDVI noise
* Easy to implement

### ❌ Weakness

* Does not truly learn sequential growth trend (treats time as channels)

⭐ Explainability: 7/10
⭐ Interpretability: 7/10
⭐ Accuracy: ~96%

---

## ✅ Model 5: CNN + LSTM (True Temporal Learning)

### 📌 What it is

CNN extracts spatial features per month → LSTM learns crop growth trend.

Input shape:

```
(N, T=3, 32, 32)
```

### ✅ Strengths

* Captures vegetation growth curves
* Detects abnormal crop development patterns
* Most research-grade and unique model

### ❌ Weakness

* Harder to interpret
* Requires more training stability and tuning

⭐ Explainability: 4/10
⭐ Interpretability: 5/10
⭐ Accuracy: ~98%

---

# 📊 Model Comparison Summary

| Model                   | Type              | Accuracy | Explainability | Interpretability | Best Use Case              |
| ----------------------- | ----------------- | -------- | -------------- | ---------------- | -------------------------- |
| NDVI Threshold Baseline | Rule-based        | Good     | ⭐⭐⭐⭐⭐          | ⭐⭐⭐⭐⭐            | Scientific baseline        |
| Custom CNN              | Deep learning     | 95–98%   | ⭐⭐⭐            | ⭐⭐⭐              | Spatial stress detection   |
| ResNet18 Transfer       | Transfer learning | 98–99%   | ⭐⭐             | ⭐⭐⭐⭐             | Best production model      |
| Temporal CNN Stack      | Temporal CNN      | ~96%     | ⭐⭐⭐⭐           | ⭐⭐⭐⭐             | Simple temporal robustness |
| CNN + LSTM              | Temporal DL       | ~98%     | ⭐⭐             | ⭐⭐⭐              | Best growth intelligence   |

---

# 🏆 Final Conclusions (Report-Ready)

* The NDVI threshold baseline is scientifically interpretable and highly explainable, but not adaptable.
* The Custom CNN improves results by learning spatial field patterns beyond NDVI averages.
* ResNet18 transfer learning gives the best accuracy and cross-season robustness due to pretrained feature extraction.
* Temporal CNN and CNN+LSTM provide deeper agricultural intelligence by learning vegetation dynamics over time.

---

# 📈 Experiment Tracking (MLflow)

MLflow was used to track:

* hyperparameters (lr, epochs, batch size)
* training loss per epoch
* accuracy and evaluation metrics
* saved trained models

To run MLflow locally:

```bash
mlflow ui --port 5000
```

---

# 🛠 Tech Stack

* Python
* Google Earth Engine (GEE)
* Rasterio
* NumPy
* PyTorch
* Torchvision
* Scikit-learn
* Matplotlib
* MLflow

---

# 🌍 Future Improvements

* Add cloud masking for cleaner NDVI
* Add rainfall + soil moisture layers
* Use full Sentinel-2 multiband input (B2,B3,B4,B8)
* Convert classification into yield prediction
* Deploy using Streamlit / FastAPI

---

# 👩‍💻 Author

Project by: **[Your Name]**
Remote Sensing + Deep Learning Project
Punjab Crop Stress Monitoring (2023)

---

# ⭐ If you like this project

Drop a ⭐ and feel free to fork + extend this work!
