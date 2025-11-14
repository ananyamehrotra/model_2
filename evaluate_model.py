import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Check which model exists
if os.path.exists("plant_disease_model_finetuned.h5"):
    model_path = "plant_disease_model_finetuned.h5"
    print("Evaluating fine-tuned model...")
elif os.path.exists("plant_disease_model.h5"):
    model_path = "plant_disease_model.h5"
    print("Evaluating base model...")
else:
    print("No trained model found!")
    exit()

# Load the model
model = load_model(model_path)

# Load validation data
val_dir = "dataset/val"
val_gen = ImageDataGenerator(rescale=1./255)

val_generator = val_gen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Evaluate the model
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

loss, accuracy = model.evaluate(val_generator, verbose=1)

print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy*100:.2f}%")
print("="*50)
