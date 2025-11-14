import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("plant_disease_model_finetuned.h5")


def predict_plant_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    class_labels =[
    "maize streak virus",
    "maize leaf spot",
    "maize leaf blight",
    "maize leaf beetle",
    "maize healthy",
    "maize grasshoper",
    "maize fall armyworm",
    "tomato verticilium wilt",
    "tomato septoria leaf spot",
    "tomato leaf curl",
    "tomato leaf blight",
    "tomato healthy",
    "cassava mosaic",
    "cassava healthy",
    "cassava green mite",
    "cassava brown spot",
    "cassava bacterial blight",
    "cashew red rust",
    "cashew leaf miner",
    "cashew healthy",
    "cashew gumosis",
    "cashew anthracnose"
]


    return class_labels[class_idx]


# Test with an image
print(predict_plant_disease(r"C:\Users\ASUS VIVOBOOK\Desktop\test_leaf.jpg"))