# 🧠 Brain Tumor Detection & Classification

An end-to-end deep learning system for automated brain tumor detection and classification using MRI scans. Built using Transfer Learning (EfficientNetB0) with Grad-CAM explainability and deployed via a Streamlit web application.

---

## 📌 Project Overview

Brain tumor diagnosis requires expert radiologists to manually examine MRI scans — a process that is time-consuming, expensive, and prone to human error. This project automates tumor detection and classification using deep learning, providing instant predictions with visual explanations to assist clinical decision-making.

**Tumor Classes:**
- 🔴 Glioma
- 🟡 Meningioma
- 🟢 Pituitary Tumor
- ⚪ No Tumor

---

## 👥 Team

| Name | Role |
|------|------|
| **Nishit Patel** | Model Building & Training |
| **Pranav Adhikari** | Model Evaluation & Metrics |
| **Unique Bhakta Shrestha** | EDA & Visualizations |
| **Pragun Lal Shrestha** | Data Preprocessing & Augmentation |

> Grad-CAM explainability and Streamlit deployment are shared across the team.

---

## 📂 Project Structure

```
brain_tumor_detection/
├── dataset/                  # MRI images (not pushed to GitHub)
│   ├── Train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Test/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb     # Preprocessing & Augmentation
│   ├── 03_model_training.ipynb    # Model Building & Training (Colab)
│   └── 04_evaluation.ipynb        # Evaluation & Grad-CAM
├── models/
│   ├── best_model_finetuned.keras # Best trained model
│   └── class_indices.json         # Class label mapping
├── app/
│   └── app.py                     # Streamlit web application
├── utils/
│   └── gradcam.py                 # Grad-CAM implementation
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Source:** Mendeley Data — Brain Tumor MRI Dataset  
**Link:** https://data.mendeley.com/datasets/zwr4ntf94j/5  
**Total Images:** 12,064 MRI scans  
**Modality:** T1-weighted contrast-enhanced MRI  

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|-------|--------|------------|----------|-----------|-------|
| Train | 3,018 | 2,183 | 1,945 | 2,504 | 9,650 |
| Test | 755 | 626 | 487 | 546 | 2,414 |

> ⚠️ Dataset is not included in this repository due to size. Download from the link above and place in the `dataset/` folder.

---

## ⚙️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| Deep Learning | TensorFlow 2.x, Keras |
| Model | EfficientNetB0 (Transfer Learning) |
| Explainability | Grad-CAM |
| Data Processing | NumPy, Pandas, OpenCV, Pillow |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn |
| Deployment | Streamlit |

---

## 🔄 Project Workflow

```
Data Collection → EDA → Preprocessing & Augmentation → Model Building
      → Training (2-Phase) → Evaluation → Grad-CAM → Streamlit Deployment
```

### 1. EDA
- Class distribution analysis
- Sample image visualization per class
- Image dimension analysis
- Pixel intensity distribution

### 2. Preprocessing & Augmentation
- Resize to 224×224 pixels
- Normalize pixel values (0–1)
- Augmentation: flip, rotate, zoom, brightness shift
- Class weights to handle imbalance
- 80/20 train/validation split

### 3. Model — EfficientNetB0 + Transfer Learning
- **Phase 1:** Frozen base, train custom head (lr=1e-3, 15 epochs)
- **Phase 2:** Fine-tune top 30 layers (lr=1e-4, 20 epochs)
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

### 4. Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report per class
- Emphasis on **Recall** to minimize false negatives

### 5. Grad-CAM Explainability
- Heatmap overlay on MRI showing regions influencing prediction
- Builds clinical interpretability and trust

### 6. Streamlit Deployment
- Upload MRI image
- Get instant prediction with confidence score
- View Grad-CAM heatmap

---

## 🚀 Setup & Run

### Prerequisites
```bash
python 3.10
```

### Installation
```bash
git clone https://github.com/yourusername/brain_tumor_detection.git
cd brain_tumor_detection
python -m venv brain_tumor_env
brain_tumor_env\Scripts\activate
pip install -r requirements.txt
```

### Download Dataset
Download from: https://data.mendeley.com/datasets/zwr4ntf94j/5  
Place the `Train/` and `Test/` folders inside `dataset/`

### Run Notebooks
Open in VS Code with Jupyter extension, run in order:
1. `01_eda.ipynb`
2. `02_preprocessing.ipynb`
3. `03_model_training.ipynb` *(run on Google Colab with T4 GPU)*
4. `04_evaluation.ipynb`

### Run Streamlit App
```bash
cd app
streamlit run app.py
```

---

## 📈 Results

> *(To be updated after training completes)*

| Metric | Score |
|--------|-------|
| Training Accuracy | - |
| Validation Accuracy | - |
| Test Accuracy | - |
| F1 Score (Weighted) | - |

---

## 🔍 Key Design Decisions

**Why EfficientNetB0?**  
Achieves high accuracy with only 5.3M parameters compared to VGG16's 138M, using compound scaling across depth, width, and resolution.

**Why Transfer Learning?**  
Our dataset of 12,064 images is insufficient to train a CNN from scratch. EfficientNetB0 pre-trained on ImageNet (1.2M images) provides rich feature representations that transfer well to medical imaging.

**Why Grad-CAM?**  
Medical AI without explainability cannot be trusted clinically. Grad-CAM makes the model's decision process transparent by highlighting the exact MRI regions that influenced the prediction.

**Why emphasize Recall?**  
A false negative (missing a real tumor) is far more dangerous than a false positive. We optimize recall to minimize the risk of undetected tumors.

---

## 📝 References

- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML 2019*
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV 2017*
- Dataset: Mendeley Data — https://data.mendeley.com/datasets/zwr4ntf94j/5

---

## 📄 License

This project is for academic purposes — 4th Semester AIML Project.
