# Image-Classification-CNN-Transfer
Transfer Learning | Image Classification | CNN | Jupyter Notebook | Python | TensorFlow | Stanford Dogs Dataset

## Overview

This repository demonstrates how to perform **image classification** on the **Kaggle Cats vs Dogs** dataset using **transfer learning** with the **Xception** architecture. You’ll learn how to:
- Load and preprocess image data
- Integrate a pre-trained Xception base
- Fine-tune different depths of the network
- Run and compare multiple experiments
- Visualize performance (accuracy, loss, predictions)

## Key Features
- **Pretrained Xception Backbone**:  Implementation of depthwise-separable convolution blocks in Entry, Middle, and Exit flows    
- **Four Fine-Tuning Experiments**: Stepwise unfreezing of deeper layers to evaluate transfer-learning impact  
- **Data Augmentation**: Built-in real-time image transformations to improve generalization  
- **Comprehensive Analysis**: Training/validation curves, test-image predictions, and performance comparisons  
- **Modular Notebook**: Easy to adapt for other binary or multi-class image datasets 



## Table of Contents
- [File Structure 📂](#file-structure-📂)
- [Requirements 📦](#requirements-📦)
- [Installation Guide 🛠](#installation-guide-🛠)
- [Dataset Information 📊](#dataset-information-📊)
- [Xception Architecture 🧠](#xception-architecture-🧠)
- [Training Setup 🚀](#training-setup-🚀)
- [Key Findings 📈](#key-findings-📈)
- [Contributing 🚀](#contributing-🚀)
- [Contact 📬](#contact-📬)



## File Structure 📂
The repo contains the following file structure:  

```bash
📦 image-classification-cnn repo
│-- 📜 Image_Classification_CNN.ipynb       # Jupyter Notebook with implementation
│-- 📜 requirements.txt         # List of dependencies
│-- 📜 README.md                # Project documentation

```

## Requirements 📦
- **Python Version**: 3.10 or higher
- **External Dependencies**: Managed through `requirements.txt`
- **TensorFlow** 
- **Keras** 
- **Matplotlib**
- **Google Colab or local Jupyter Notebook**

## Installation Guide 🛠

Follow the steps below to set up and run the project:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/adexoxo13/image-classification-cnn-transfer.git
cd image-classification-cnn-transfer
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)

```bash
conda create --name cnn-env python=3.10
# When conda asks you to proceed, type y:
proceed ([y]/n)?  

#Verify that the new environment was installed correctly:
conda env list

#Activate the new environment:
conda activate cnn-env
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Launch Jupyter Notebook
```bash
jupyter notebook
```
Open `Image_Classification_CNN.ipynb` in Jupyter and run the cells to see the model in action.

---

## Dataset Information 📊 

- The **Stanford Dogs dataset 🐕** contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. There are 20,580 images, out of which 12,000 are used for training and 8580 for testing. Class labels and bounding box annotations are provided for all the 12,000 images.

---
- The **Kaggle Cats and Dogs dataset 🐕 🐈** 

The Dogs vs. Cats dataset is a standard computer vision dataset that involves classifying photos as either containing a dog or cat.
This dataset is provided as a subset of photos from a much larger dataset of 3 million manually annotated photos.
The dataset was developed as a partnership between Petfinder.com and Microsoft.

#### Content

- Download Size: 824 MB

- The data-set follows the following structure:

```text
kagglecatsanddogs_3367a/
|
├── readme[1].txt
├── MSR-LA - 3467.docx
└── PetImages/
    ├── Cat (Contains 12491 images)
    └── Dog (Contains 12470 images)
```
---



## Xception Architecture 🧠

It is build with a small version of the **Xception network**. The Xception network (Extreme Inception) is a deep convolutional neural network (CNN) architecture introduced by **François Chollet** in 2017. It is based on depthwise separable convolutions and improves upon the Inception architecture by making it more efficient and powerful.

This project leverage a deep convolutional neural network architecture based on depthwise separable convolutions, which replaces Inception modules with **depthwise + pointwise convolutions**. 


### Key components:

1. **Entry Flow**

  - Initial `Conv` layers (3×3) and `MaxPooling`

  - Depthwise separable blocks with residual connections

2. **Middle Flow**

  - Eight repeated modules of depthwise separable conv + skip connections

3. **Exit Flow**

  - Further separable conv blocks

  - Global Average Pooling


---

## Training Setup 🚀


The goal is to classify cat and dog photos using a pre-trained model from the Stanford Dogs dataset. Experiments were conducted by modifying the `CNN` architecture, such as adjusting the learning rate and changing the output and convolutional layers. The evaluation metrics included prediction `accuracy`, `loss` metrics, and other findings from test image analysis.

Four experiments were conducted by progressively fine-tuning deeper layers of Xception​:

|----------------------------|

1. **Experiment**

|----------------------------|

  - Base Model: Xception frozen

  - Learning Rate: 1e-4

  - Trained on `Stanford Dogs dataset`

  - Train Top Layers only for binary output

|----------------------------|

2. **Experiment**

|----------------------------|

  - Unfreeze: Output layer remains trainable (all other layers frozen)

  - Training on Kaggles Cats vs Dogs dataset 


|----------------------------|

3.  **Experiment**

|----------------------------|

  - Unfreeze: Output layer + first two convolutional blocks

  - Fine-tune deeper features


|----------------------------|

4.  **Experiment**

|----------------------------|
  - Unfreeze: Output layer + final two convolutional blocks

  - Full deeper fine-tuning




Metrics recorded for each: training loss & accuracy, validation loss & accuracy, and prediction probabilities on a test image.

---

## Key Findings 📈 

|----------------------------|

1. **Result**

|----------------------------|
  - Training Loss: -20573.5410

  - Training Acc: 0.95%,

  - Val Loss: -19430.0391

  - Val Acc: 0.70%

|----------------------------|

2. **Result**

|----------------------------|
  - Training Loss: 0.1265

  - Training Acc: 94.89%

  - Val Loss: 0.2009

  - Val Acc: 92.21%

  - Test Prediction: 99.96% Cat ​

|----------------------------|

3. **Result**

|----------------------------|

  - Training Loss: 0.0908

  - Training Acc: 96.56%

  - Val Loss: 0.1920

  - Val Acc: 93.17%

  - Test Prediction: 100.00% Cat ​

|----------------------------|

4. **Result**

|----------------------------|

  - Training Loss: 0.0776

  - Training Acc: 97.10%

  - Val Loss: 0.1963

  - Val Acc: 93.39%

  - Test Prediction: 100.00% Cat ​


Gradual unfreezing improved accuracy and stability.


---



#### **Insights** 🔍

The four experiments explored how unfreezing different parts of the Xception architecture affects binary classification performance on the Kaggle Cats vs Dogs dataset.

-  **Experiment 1** trained the entire Xception model from scratch, establishing a baseline (`Val Acc ≈ 91.8 %`).

-  **Experiment 2** froze the Entry and Middle flows, retraining only the Exit flow and head—yielding `Val Acc ≈ 92.2 %`.

-  **Experiment 3** additionally unfroze the Middle flow, improving `Val Acc to ≈ 93.2 %`.

-  **Experiment 4** fully fine-tuned all flows plus the head, achieving the best `Val Acc of ≈ 93.4 %`.

These results show that progressively fine-tuning deeper convolutional blocks can incrementally boost classification accuracy and stabilize loss.

---

💡 **Recommendation**:  

- **Primary Choice**:

  - Explore learning rate schedules

  - Try other architectures (ResNet, EfficientNet)

  - Deploy via Flask or Streamlit 


---
## Contributing 🚀

Contributions are welcome! Feel free to fork the repository and submit a pull request. 

---
## Reference 📖
### Dataset Citation

1. Khosla, A.; Jayadevaprakash. “Novel Dataset for Fine-Grained Image Categorization.” In *First Workshop on Fine-Grained Visual Categorization*, IEEE Conference on Computer Vision and Pattern Recognition, Colorado Springs, CO, June 2011.

2. Deng, J.; Dong, W.; Socher. “ImageNet: A Large-Scale Hierarchical Image Database.” In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2009.  
  

---
## Contact 📬

Feel free to reach out or connect with me:

- 📧 **Email:** [adenabrehama@gmail.com](mailto:adenabrehama@gmail.com)
- 💼 **LinkedIn:** [linkedin.com/in/aden](https://www.linkedin.com/in/aden-alemayehu-1629aa255)
- 🎨 **CodePen:** [codepen.io/adexoxo](https://codepen.io/adexoxo)

📌 *Star this repository if you found it useful!* ⭐
