import os

train_path = r"C:\Users\ASUS VIVOBOOK\Downloads\Dataset for Crop Pest and Disease Detection\Dataset for Crop Pest and Disease Detection\CCMT Dataset-Augmented\Maize\train_set"  # <- Change to your train folder path

classes = []
for folder in os.listdir(train_path):
    folder_path = os.path.join(train_path, folder)
    if os.path.isdir(folder_path):
        count = len(os.listdir(folder_path))
        classes.append((folder, count))

# Sort by number of images per class
classes.sort(key=lambda x: x[1], reverse=True)

for label, count in classes:
    print(f"{label}: {count}")