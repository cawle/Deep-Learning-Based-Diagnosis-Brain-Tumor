# Deep Learning-Based Brain Tumor Diagnosis

## 📌 Project Overview
This project presents a deep learning approach for **automated brain tumor classification** using MRI scans. The model is built on the **Xception architecture** with transfer learning and data augmentation techniques to classify images into four categories:

- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

## 🧠 Key Features
- **Transfer Learning**: Utilizes Xception model pre-trained on ImageNet.
- **Data Augmentation**: Includes rotation, zoom, shear, brightness adjustment, and flipping to enhance generalization.
- **Two-Phase Training**:
  1. **Feature Extraction** – frozen base layers.
  2. **Fine-tuning** – unfrozen entire model with a lower learning rate.
- **Class Imbalance Handling**: Dynamic class weighting and regularization via dropout.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.

## 📊 Results
- **Test Accuracy**: **72.08%**
- **Strong Performance**: Meningioma and No Tumor classes.
- **Challenges**: Glioma classification showed higher misclassification rates.
- **Training Stability**: Minimal overfitting, consistent validation loss.

## 🛠️ Technologies Used
- **Programming Language**: Python 3.10
- **Deep Learning Framework**: TensorFlow, Keras
- **Data Source**: Kaggle Brain Tumor MRI Dataset
- **Visualization Tools**: Matplotlib, Seaborn, Grad-CAM (for interpretability)
- **Development Tools**: Jupyter Notebook, Google Colab

## 📁 Dataset
- **Source**: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **Preprocessing**: Resized to 299×299 pixels, normalized to [0,1] range.
- **Augmentation**: Applied to training set only.

## 🧪 Methodology
1. **Data Loading & Preprocessing**
2. **Augmentation** (Rotation, Zoom, Shear, Brightness, Flip)
3. **Model Building** (Xception + custom dense layers)
4. **Two-Phase Training**
5. **Evaluation & Visualization**

## 📈 Model Architecture
Xception Base (frozen initially)  
↓  
GlobalAveragePooling2D  
↓  
Dense (1024, ReLU)  
↓  
Dropout (0.5)  
↓  
Dense (4, Softmax)

---

## text

---

## ✅ How to Run
1. Clone the repository.  
2. Install dependencies:  
   pip install tensorflow numpy pandas matplotlib scikit-learn  
3. Download the dataset from Kaggle.  
4. Organize the dataset into train/ and test/ folders.  
5. Run the Jupyter notebook or Python script for training and evaluation.

---

## 🔍 Interpretability
- Confusion Matrix for performance breakdown  
- Training/Validation plots for learning behavior  
- Sample predictions with true vs. predicted labels  

---

## 🚧 Limitations & Future Work

### Limitations
- Dataset size is limited and imbalanced  
- Image quality varies due to MRI resolution  
- Interpretability is limited  

### Future Work
- Integrate Grad-CAM or SHAP for clinical trust  
- Use larger and more diverse datasets  
- Hybrid or ensemble models  
- Real-time deployment for clinical use  

---

## 📚 References
All cited works are listed in the paper (Pages 13–16).  
Key references include studies on CNN-based tumor classification, transfer learning, and data augmentation in medical imaging.

---

## 👥 Authors
**Abdullahi Mohamed Ali**  
abdullahi.ali@student.aiu.edu.my  

**Anas Ismail Ahmed**  
anasismail.ahmed@student.aiu.edu.my  

School of Computer and Informatics  
Albulkhary International University, Malaysia
