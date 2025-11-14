# fine_tune_model.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# Load previous model
model = load_model("plant_disease_model.h5")

# Re-load dataset
train_dir = "dataset/train"
val_dir = "dataset/val"

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical")
val_generator = val_gen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode="categorical")

# Unfreeze last 30 layers of base model
model.layers[0].trainable = True
for layer in model.layers[0].layers[:-30]:
    layer.trainable = False

# Compile and fine-tune
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

model.save("plant_disease_model_finetuned.h5")

# Save Fine-tuning History and Display Results
import json

history_dict = {
    'train_accuracy': [float(acc) for acc in history.history['accuracy']],
    'val_accuracy': [float(acc) for acc in history.history['val_accuracy']],
    'train_loss': [float(loss) for loss in history.history['loss']],
    'val_loss': [float(loss) for loss in history.history['val_loss']]
}

with open("finetuning_history.json", "w") as f:
    json.dump(history_dict, f, indent=4)

print("\n" + "="*60)
print("FINE-TUNING COMPLETED!")
print("="*60)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print("="*60)
print("Fine-tuning history saved to: finetuning_history.json")
print("="*60)
