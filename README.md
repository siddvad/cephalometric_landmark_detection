# Cephalometric Landmark Detection

An automated deep learning application for detecting **29 cephalometric landmarks** from lateral cephalometric X-ray images.

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cephalometriclandmarkdetection-jnksxeomm9vgu59mxwklbm.streamlit.app/)

## 📌 Overview

Cephalometric landmark detection is an important step in orthodontic and craniofacial analysis. This project uses a **U-Net based convolutional neural network** to automatically locate anatomical landmarks on lateral cephalometric X-rays.

The application accepts an X-ray image, preprocesses it, predicts a heatmap for each landmark, extracts the corresponding coordinates, and displays the detected landmarks directly on the original image.

## ✨ Features

- Automatic detection of **29 cephalometric landmarks**
- U-Net based landmark detection architecture
- Grayscale conversion and image normalization
- CLAHE-based contrast enhancement
- Heatmap-based landmark prediction
- Mapping of predicted coordinates back to the original X-ray
- Visualization of detected landmarks on the X-ray
- Display of landmark names and predicted coordinates
- Interactive web interface built with Streamlit

## 🧠 Model

The model is a custom U-Net architecture designed for landmark heatmap prediction.

| Parameter | Value |
|---|---|
| Architecture | U-Net |
| Input | `1 × 512 × 512` |
| Output | `29 × 128 × 128` |
| Input channels | 1 |
| Number of landmarks | 29 |
| Output representation | Landmark heatmaps |

Each output channel corresponds to one anatomical landmark. The predicted landmark position is obtained from the maximum activation in the corresponding heatmap.

## 🔄 Processing Pipeline

```text
Input X-ray
     ↓
Grayscale Conversion
     ↓
CLAHE Contrast Enhancement
     ↓
Image Normalization
     ↓
Resize to 512 × 512
     ↓
U-Net Model
     ↓
29 Landmark Heatmaps
     ↓
Coordinate Extraction
     ↓
Map Coordinates to Original Image
     ↓
Landmark Visualization