import json
import os

print("\n" + "="*60)
print("MODEL TRAINING HISTORY VIEWER")
print("="*60)

# Check for training history files
files_to_check = [
    ("training_history.json", "Initial Training"),
    ("finetuning_history.json", "Fine-tuning")
]

found_any = False

for filename, stage_name in files_to_check:
    if os.path.exists(filename):
        found_any = True
        with open(filename, 'r') as f:
            history = json.load(f)
        
        print(f"\n{stage_name} Results:")
        print("-" * 60)
        print(f"Final Training Accuracy: {history['train_accuracy'][-1]*100:.2f}%")
        print(f"Final Validation Accuracy: {history['val_accuracy'][-1]*100:.2f}%")
        print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
        print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
        print(f"\nEpochs trained: {len(history['train_accuracy'])}")
        print("\nEpoch-by-epoch Validation Accuracy:")
        for i, acc in enumerate(history['val_accuracy'], 1):
            print(f"  Epoch {i}: {acc*100:.2f}%")

if not found_any:
    print("\n❌ No training history files found.")
    print("\nTo get accuracy information, you need to:")
    print("1. Re-run the training with the updated scripts, OR")
    print("2. Check your terminal output from when you trained the model")
    print("\nThe models exist but without history files:")
    if os.path.exists("plant_disease_model.h5"):
        print("  ✓ plant_disease_model.h5 (initial model)")
    if os.path.exists("plant_disease_model_finetuned.h5"):
        print("  ✓ plant_disease_model_finetuned.h5 (fine-tuned model)")

print("\n" + "="*60)
