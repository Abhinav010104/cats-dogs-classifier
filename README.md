# Cats vs Dogs CNN Classification

A deep learning project using **transfer learning** with MobileNetV2 to classify images as cats or dogs. Achieves high accuracy through fine-tuning a pretrained ImageNet model on the curated cats-and-dogs dataset.

## 📊 Project Overview

- **Model**: MobileNetV2 (pretrained on ImageNet)
- **Dataset**: Google's curated cats-and-dogs filtered dataset (~2,000 images)
- **Task**: Binary classification (Cat / Dog)
- **Approach**: Transfer Learning + Fine-tuning
- **Input Size**: 224×224 RGB images
- **Performance**: High accuracy & precision (see results below)

## 🎯 Key Features

✅ **Transfer Learning** — Leverages pretrained MobileNetV2 weights for faster convergence and better accuracy  
✅ **Data Augmentation** — Random flips, rotations, and zoom to improve generalization  
✅ **Two-Stage Training** — Head training (frozen base) + fine-tuning (unfrozen deeper layers)  
✅ **Regularization** — Dropout layers to reduce overfitting  
✅ **Model Export** — Saves in both HDF5 (.h5) and Keras (.keras) formats  
✅ **Visualization** — Plots training/validation accuracy and loss curves  

## 📁 Project Structure

```
cats-dogs-cnn/
├── train_optimized.py     # Transfer learning with MobileNetV2
├── train.py               # Alternative training script
├── app.py                 # Streamlit web interface for predictions
├── test_tfds.py          # Dataset loading example using TensorFlow Datasets
├── models/
│   ├── best_model.h5     # Saved model (HDF5 format)
│   └── best_model.keras  # Saved model (Keras format)
├── artifacts/            # Training outputs & logs
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/cats-dogs-cnn.git
   cd cats-dogs-cnn
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model

**Train the Transfer Learning Model**
```bash
python train_optimized.py
```
This trains MobileNetV2 with a frozen base layer, then fine-tunes deeper layers with a low learning rate to achieve optimal accuracy.

### Making Predictions

**Via Streamlit Web App**
```bash
streamlit run app.py
```
Upload an image to get instant predictions with confidence scores.

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **98%** |
| **Precision** | **97%** |
| **Training Time** | ~5 min |
| **Model Size** | ~89 MB |
| **Approach** | Transfer Learning (MobileNetV2) |

**Why this approach excels:**
- Pretrained ImageNet features capture universal visual patterns (edges, textures, shapes)
- Rapid convergence with frozen base + fine-tuning strategy
- Excellent generalization on limited datasets (~2,000 images)
- Minimal overfitting due to dropout regularization and data augmentation

## 💡 Key Insights

1. **Transfer Learning > Training from Scratch** for small datasets
2. **Input size matters** — MobileNetV2 expects 224×224 inputs
3. **Fine-tuning strategy** — Freeze early layers, unfreeze deeper layers with low LR
4. **Data augmentation is crucial** — Reduces overfitting on ~2K images
5. **Dropout regularization** — Essential when unfreezing many layers

## 🔧 Technologies Used

- **TensorFlow / Keras** — Deep learning framework
- **NumPy** — Numerical computing
- **Matplotlib** — Visualization
- **Streamlit** — Web interface
- **Python 3.8+** — Programming language

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Created as a deep learning portfolio project demonstrating best practices in computer vision and transfer learning.

---

**Questions or improvements?** Feel free to open an issue or submit a pull request!
