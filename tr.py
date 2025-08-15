import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = './data'
DEEPFAKE_DIR = os.path.join(DATA_DIR, 'deepfake')
ORIGINAL_DIR = os.path.join(DATA_DIR, 'original')
MODEL_PATH = 'deepfake_detector.h5'
LOG_DIR = 'logs'def load_images():
    images, labels = [], []
for img_name in os.listdir(DEEPFAKE_DIR):
    img_path = os.path.join(DEEPFAKE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(1) 

for img_name in os.listdir(ORIGINAL_DIR):
    img_path = os.path.join(ORIGINAL_DIR, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(0) 

return np.array(images), np.array(labels)def preprocess_data(images, labels):
    images = images.astype('float32') / 255.0
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

return X_train, X_val, X_test, y_train, y_val, y_testdef create_data_generators():
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )val_datagen = ImageDataGenerator()

return train_datagen, val_datagendef build_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))# Fine-tune top layers
for layer in base_model.layers[:100]:
    layer.trainable = False
for layer in base_model.layers[100:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=1e-4),
             loss='binary_crossentropy',
             metrics=['accuracy'])

return modeldef train_model(model, train_datagen, val_datagen, X_train, X_val, y_train, y_val):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_dir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, tensorboard_callback]
)

return historydef evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)accuracy = accuracy_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

return y_pred_binarydef visualize_predictions(X_test, y_test, y_pred, num_samples=5):
    indices = np.random.choice(len(X_test), num_samples, replace=False)plt.figure(figsize=(15, 5))
for i, idx in enumerate(indices):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(X_test[idx])
    plt.title(f"True: {'Deepfake' if y_test[idx] == 1 else 'Original'}\n"
             f"Pred: {'Deepfake' if y_pred[idx] == 1 else 'Original'}")
    plt.axis('off')
plt.savefig('predictions.png')
plt.close()def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Model")
    parser.add_argument('--train', action='store_true', help='Train the model')
    args = parser.parse_args()if args.train:
    images, labels = load_images()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(images, labels)
    
    train_datagen, val_datagen = create_data_generators()
    
    model = build_model()
    history = train_model(model, train_datagen, val_datagen, X_train, X_val, y_train, y_val)
    
    model.save(MODEL_PATH)
    
    y_pred = evaluate_model(model, X_test, y_test)
    
    visualize_predictions(X_test, y_test, y_pred)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.close()if __name__ == "__main__":
    main()

