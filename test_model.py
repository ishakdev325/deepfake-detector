import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse

IMG_SIZE = (299, 299)
MODEL_PATH = 'deepfake_detector.h5'

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image")
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(model, image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)[0][0]
    return "Deepfake" if prediction > 0.5 else "Original", prediction

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Prediction")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()

    model = load_model(MODEL_PATH)

    label, confidence = predict_image(model, args.image)
    print(f"Prediction: {label} (Confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()