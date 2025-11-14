# train_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
import json

# 1. Load Data
train_dir = "dataset/train"
val_dir = "dataset/val"

train_gen =ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)
val_gen = ImageDataGenerator(
    rescale=1.0/255,

)

train_generator = train_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

val_generator = val_gen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# 2. Build Base Model
base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# 3. Compile and Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15
)

# 4. Save Model & Class Labels
model.save("plant_disease_model.h5")

with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

# 5. Save Training History and Display Results
history_dict = {
    'train_accuracy': [float(acc) for acc in history.history['accuracy']],
    'val_accuracy': [float(acc) for acc in history.history['val_accuracy']],
    'train_loss': [float(loss) for loss in history.history['loss']],
    'val_loss': [float(loss) for loss in history.history['val_loss']]
}

with open("training_history.json", "w") as f:
    json.dump(history_dict, f, indent=4)

print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print("="*60)
print("Training history saved to: training_history.json")
print("="*60)
