# 🌾 Crop Stress Detection using NDVI & Deep Learning (Punjab, India)

This repository contains an end-to-end **Remote Sensing + Deep Learning pipeline** for detecting crop vegetation stress using **Sentinel-2 satellite imagery** from Punjab, India.

The project covers:
- Satellite data extraction using **Google Earth Engine (GEE)**
- NDVI computation and patch dataset creation
- Training multiple ML/DL models:
  - NDVI Threshold Baseline
  - Custom CNN
  - ResNet18 Transfer Learning
  - Temporal NDVI CNN
  - CNN + LSTM Temporal Model
- Comparing models using **Accuracy, Explainability, Interpretability**
- Experiment tracking using **MLflow**

---

## 🎯 Project Motivation

Agricultural monitoring requires detecting crop health/stress early and accurately.

Satellite vegetation indices like **NDVI** provide a strong scientific signal of plant health, but deep learning models can go beyond simple NDVI thresholds by learning:
- spatial crop texture
- field structure
- seasonal variation
- temporal growth patterns

---

## 🛰 Dataset (Not Uploaded to GitHub)

⚠️ The dataset files (`.tif`) are **very large GeoTIFF files**, so they are **NOT included** in this GitHub repository.

Instead, this README explains **how to reproduce the dataset** from scratch using Google Earth Engine.

---

# 📌 DATA COLLECTION PIPELINE (Google Earth Engine)

This dataset was created using **Sentinel-2 Surface Reflectance** imagery.

---

## 🌍 Data Source

- **Sentinel-2 SR (Surface Reflectance)**
- Google Earth Engine collection:
  - `COPERNICUS/S2_SR_HARMONIZED`

---

## 🗺 Study Region

- Punjab, India (cropland belt)
- Region selected using polygon geometry in GEE

---

## 📅 Seasons Used

### 🌾 Rabi Season (Wheat)
- Example time range: Nov 2022 – Mar 2023

### 🌱 Kharif Season (Rice / Monsoon Crops)
- Example time range: Jun 2023 – Oct 2023

---

## 🧠 Bands Exported

Sentinel-2 bands used:

| Band | Name | Use |
|------|------|-----|
| B2 | Blue | RGB visualization |
| B3 | Green | RGB visualization |
| B4 | Red | NDVI |
| B8 | NIR | NDVI |

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
