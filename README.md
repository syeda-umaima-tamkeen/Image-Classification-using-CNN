# Image Classification Using Convolutional Neural Networks (CNN)

**Project by:** Syeda Umaima Tamkeen

## 📝 Summary

In this project, I developed a deep learning model using a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset** into 10 categories (e.g., airplanes, cats, cars, ships, etc.). I applied **data augmentation**, **regularization**, and **fine-tuning** techniques to improve generalization and reduce overfitting.

The final model achieved an accuracy of **82.3%**, demonstrating the power and effectiveness of CNNs in solving image classification problems.

---

## 🔍 Key Steps

### 1. Data Exploration
- Used the **CIFAR-10 dataset** with 60,000 32x32 color images in 10 categories.
- Visualized sample images to understand class distribution and variety.
- Verified class balance to ensure fair model training.

### 2. Data Preprocessing & Augmentation
- **Normalization:** Scaled pixel values between 0 and 1.
- **Augmentation:** Applied random transformations (rotation, shift, flip) to prevent overfitting and increase dataset diversity.

### 3. Building the CNN Architecture
- **3 Convolutional Layers** with ReLU activation and MaxPooling.
- **Fully Connected Layer** followed by a **Dropout layer** for regularization.
- **Output Layer**: 10 neurons with Softmax activation (multi-class classification).

### 4. Model Training
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Data Split:** 80% training, 20% testing  
- **Epochs:** 15  
- **Batch Size:** 64  
- Trained with data augmentation for better generalization.

### 5. Model Evaluation
- Evaluated on accuracy, precision, recall, and F1-score per class.
- Plotted training vs. validation accuracy to monitor learning progress.
- **Accuracy Achieved:**  
  - **Training Accuracy:** 82.3%  
  - **Validation Accuracy:** 79.6%

### 6. Model Fine-Tuning
- **Dropout (0.5):** Reduced overfitting.
- **Early Stopping:** Halted training when validation performance stopped improving.
- Tuned hyperparameters like learning rate and number of filters.

---

## ✅ Results

### 📈 Evaluation Metrics
- **Final Accuracy:** 82.3%
- **Validation Accuracy:** 79.6%
- **Loss:** Reduced with regularization and early stopping.
- **Precision & Recall:** Strong performance across most categories (especially airplane, cat, ship).

### 🔧 Best Hyperparameters
- **Epochs:** 15  
- **Batch Size:** 64  
- **Dropout Rate:** 0.5  
- **Optimizer:** Adam (with learning rate tuning)

---

## 📌 Conclusion

The CNN model trained on the CIFAR-10 dataset achieved strong performance in classifying diverse image categories. Techniques like **data augmentation**, **Dropout**, and **early stopping** improved the model's generalization.

### 🔭 Future Enhancements:
- Implement **Transfer Learning** using pre-trained models like VGG or ResNet.
- Try **deeper CNN architectures**.
- Tune more hyperparameters with **RandomizedSearchCV** or **Optuna**.

This project showcases the power of CNNs in computer vision and forms a solid foundation for advanced work in image classification.
