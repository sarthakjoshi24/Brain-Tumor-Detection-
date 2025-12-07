

🧠 Brain Tumor Detection Using CNN

This project implements a binary image classification model using a Convolutional Neural Network (CNN) to detect the presence of brain tumors from MRI images. The system efficiently preprocesses MRI scans, trains a robust CNN model, evaluates performance, and optionally deploys predictions through a Streamlit application.

📂 Project Structure
├── __pycache__/
├── .vscode/
├── archive/                     # Archived files / old experiments
├── Br35H-Mask-RCNN/             # Separate exploration folder (Mask-RCNN Experiments)
├── Brain_tumour_detecti.../     # Additional reference dataset/model exploration
├── Datasets/
│   ├── no/                      # MRI images without tumor
│   ├── yes/                     # MRI images with tumor
│   └── pred/                    # Images for model predictions
├── venv/                        # Virtual environment (ignored in Git)
├── app.py                       # Streamlit deployment script
├── archive.zip                  # Compressed old resources
├── BrainTumor10Epochs...        # Model checkpoint files
├── BrainTumor10Epochs...        # (another model checkpoint file)
├── maintest1.py                 # Script for testing model predictions
├── maintrain1.py                # Main training script for CNN model
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies

🚀 Project Overview
This project uses deep learning to classify MRI brain scans as:
Yes (Tumor Present)
No (Tumor Not Present)

It uses TensorFlow/Keras to build and train a CNN model capable of achieving high classification accuracy. The preprocessing pipeline ensures the images are normalized, cleaned, and ready for training. Deployment is optionally done via Streamlit for easy real-time predictions.
<img width="702" height="496" alt="Screenshot 2025-11-26 015347" src="https://github.com/user-attachments/assets/93159ea4-2463-4c11-93f1-834622dd330e" />

🛠️ Implementation Workflow

<img width="830" height="217" alt="Screenshot 2025-11-26 014658" src="https://github.com/user-attachments/assets/c345ddf0-5481-469e-b138-a4af88567c61" />

1. Data Collection

MRI images sourced from publicly available datasets and organized into yes/ and no/ folders under Datasets/.

2. Data Preprocessing
-Performed inside training/testing scripts:
-Resizing and normalization
-Grayscale or RGB conversions
-Train-test split
-Optional histogram equalization

3. Exploratory Data Analysis (EDA)
-Basic visualization and dataset inspection to understand distribution, sample images, and class balance.

4. Model Design
-Custom CNN architecture implemented in maintrain1.py:
-Convolution + MaxPooling layers
-Dense fully connected layers
-Dropout to reduce overfitting
-Sigmoid activation for binary classification

5. Model Training
->maintrain1.py handles:
-Training for defined epochs
-Accuracy/loss monitoring
-Saving trained model (BrainTumor10Epochs.h5 etc.)

6. Model Evaluation
-Accuracy scores
-Loss curves
-Confusion matrix
Example predictions (in maintest1.py)

7. Hyperparameter Tuning
-Manual tuning includes:
-Batch size
-Learning rate
-Number of filters/layer depth
-Epoch adjustments

8. Prediction & Insights
maintest1.py loads the trained model and predicts tumor presence on images inside the pred/ folder.

9. Deployment (Optional)
i)app.py provides a Streamlit web interface:
ii)Upload MRI images
iii)See prediction result with confidence score

🧩 Tech Stack
Language
Python

Libraries
TensorFlow / Keras
NumPy
Pandas
OpenCV
Matplotlib
Streamlit

🎯 Results


<img width="1723" height="606" alt="homepage3" src="https://github.com/user-attachments/assets/32c5e261-db3b-48e6-9c1a-783879a835c9" />
<img width="1404" height="669" alt="upload2" src="https://github.com/user-attachments/assets/b8992469-6abe-4c09-a5eb-cc9c7478fda0" />
<img width="1545" height="931" alt="prediction" src="https://github.com/user-attachments/assets/c9478802-0a01-4f92-88e4-f5427a6bb97a" />

-Achieved high accuracy in classifying MRI images
-Robust CNN architecture
-Performs well on unseen MRI images
-Ready for demo using Streamlit

🙌 Acknowledgements
Datasets used from publicly available MRI repositories.
Kaggle dataset Link:https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection
Thanks to the open-source community for tools and support.


