Sign Language Recognition System

This project is a real-time Sign Language Recognition system that uses a Convolutional Neural Network (CNN) model trained on the ASL Alphabet dataset to recognize hand gestures from a webcam feed.

📁 Project Structure

sign_language_project/
├── dataset/
│   └── asl_alphabet_train/   # A–Z folders with hand sign images
├── model/
│   └── sign_model.h5         # Trained CNN model
├── train_model.py            # Script to train the model
├── main.py                   # Real-time prediction using webcam
├── requirements.txt          # List of dependencies
└── README.md                 # Project overview


Features:
- Detects American Sign Language (A–Z) in real-time using webcam
- Uses a custom-trained CNN model
- Predicts and displays the detected sign with confidence
- Fast and efficient model with option to improve accuracy
- Modular code (training and inference separated)


Requirements:
Install required Python packages using:
pip install -r requirements.txt

Dependencies include:
	•	tensorflow
	•	opencv-python
	•	numpy

How It Works? 
Training:
	•	Run train_model.py to train the model on dataset/asl_alphabet_train/
	•	Model is saved to model/sign_model.h5
Real-Time Prediction:
	•	Run main.py to start webcam-based detection
	•	A green box will appear — show your sign inside that box
	•	The model will predict the sign and display it with confidence
 
Train the model:
python train_model.py

Run real-time detection:
python main.py
