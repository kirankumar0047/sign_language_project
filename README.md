# Sign Language Recognition System

![Developer](https://img.shields.io/badge/Developed%20By-Kolla%20Kiran%20Kumar-red)

A real-time **Sign Language Recognition System** that uses a **CNN model** trained on the ASL Alphabet dataset to recognize hand gestures through a webcam.

## Features

- Real-time ASL (A–Z) recognition using a webcam.
- CNN-based deep learning model for gesture classification.
- Displays predicted sign with confidence.
- Modular implementation for training and real-time prediction.

## Tech Stack

- Python
- TensorFlow/Keras
- OpenCV
- NumPy

## Project Structure

```text
sign_language_project/
├── dataset/
│   └── asl_alphabet_train/      # A–Z hand sign images
├── model/
│   └── sign_model.h5            # Trained CNN model
├── train_model.py               # Model training script
├── main.py                      # Real-time prediction
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python train_model.py
python main.py
```
