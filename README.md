# DICOM Viewer (Python, PyQt)

A Python-based DICOM viewer built with PyQt, providing medical image visualization
and essential image manipulation features.

This project was developed in **2023** as part of an industry–academia collaboration project**,
focusing on understanding medical imaging data and building a GUI-based visualization tool.

---

## 🧠 Motivation

DICOM (Digital Imaging and Communications in Medicine) is the standard format for
medical imaging data used in clinical environments.
However, handling DICOM files requires both domain knowledge (medical imaging)
and technical skills (image processing and GUI development).

This project aims to:
- Understand the structure of DICOM files (pixel data and metadata)
- Implement a functional DICOM viewer from scratch
- Gain hands-on experience with Python-based GUI development

---

## ✨ Features

- Load and visualize DICOM (.dcm) files
- Display patient metadata using DICOM tags
- 2D medical image display
- Image manipulation tools commonly used in clinical settings:
  - Brightness adjustment
  - Image flipping (horizontal / vertical)
  - Image rotation
  - Zoom in / zoom out
  - Image filtering (e.g. blur)

---

## 🛠 Tech Stack

- **Language**: Python  
- **GUI**: PyQt  
- **Medical Imaging**: pydicom  
- **Image Processing**: NumPy, OpenCV

---
▶️ How to Run
# 1. Clone the repository
git clone https://github.com/USERNAME/dicom-viewer.git
cd dicom-viewer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app/main.py
