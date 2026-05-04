# =========================================
# 🌱 SmartHarvest AI - Final Clean Version
# =========================================

# ==============================
# 1. SETUP
# ==============================
from google.colab import drive
drive.mount('/content/drive')

import os
import zipfile
import shutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image

# ==============================
# 2. DATA PREPARATION
# ==============================
base_path = "/content/dataset"

# Clean old dataset
if os.path.exists(base_path):
    shutil.rmtree(base_path)

os.makedirs(base_path, exist_ok=True)

zip_files = {
    "train": "/content/drive/MyDrive/train.zip",
    "validation": "/content/drive/MyDrive/validation.zip",
    "test": "/content/drive/MyDrive/test.zip"
}

# Extract dataset
for key, path in zip_files.items():
    extract_path = os.path.join(base_path, key)
    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

print("Dataset Ready")

# ==============================
# 3. LOAD DATA
# ==============================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{base_path}/train",
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{base_path}/validation",
    image_size=(224, 224),
    batch_size=32
)

class_names = train_ds.class_names
print("Classes:", class_names)

# ==============================
# 4. BUILD MODEL
# ==============================
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),

    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# 5. TRAIN MODEL
# ==============================
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ==============================
# 6. SAVE MODEL
# ==============================
model.save('/content/drive/MyDrive/fruit_model.keras')
print("Model Saved")

# ==============================
# 7. LOAD MODEL
# ==============================
from tensorflow.keras.models import load_model
model = load_model('/content/drive/MyDrive/fruit_model.keras')

# ==============================
# 8. CATEGORY LIST
# ==============================
fruit_classes = ['apple','banana','grapes','mango','orange','watermelon','kiwi','lemon','pear','pineapple','pomegranate']

vegetable_classes = ['beetroot','bell pepper','carrot','cabbage','capsicum','cauliflower','chili pepper','corn','eggplant','lettuce','potato','tomato','onion','cucumber','radish','spinach','sweet potato']

spices_classes = ['garlic','ginger','jalapeno','paprika']

legumes_classes = ['peas','soy beans']

# ==============================
# 9. PREDICTION
# ==============================
from google.colab import files

uploaded = files.upload()
img_path = list(uploaded.keys())[0]

# Load image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

threshold = 0.75

# Category logic
if confidence < threshold:
    title = f"Unknown ❌\nConfidence: {confidence*100:.2f}%"
else:
    if predicted_class in fruit_classes:
        category = "FRUIT 🍎"
    elif predicted_class in vegetable_classes:
        category = "VEGETABLE 🥦"
    elif predicted_class in spices_classes:
        category = "SPICE 🌶️"
    elif predicted_class in legumes_classes:
        category = "LEGUME 🌱"
    else:
        category = "OTHER"

    title = f"{category}\n{predicted_class} ({confidence*100:.2f}%)"

# Display result
plt.imshow(img)
plt.title(title)
plt.axis('off')
plt.show()

print("Prediction:", predicted_class)
print("Confidence:", f"{confidence*100:.2f}%")
