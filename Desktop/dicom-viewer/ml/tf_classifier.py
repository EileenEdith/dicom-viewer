import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

CLASS_LABELS = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]

class TumorClassifierTF:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = (224, 224)

    def preprocess(self, img_path):
        img = image.load_img(img_path, target_size=self.img_size)
        x = image.img_to_array(img)
        return np.expand_dims(x, axis=0)

    def predict(self, img_path):
        x = self.preprocess(img_path)
        preds = self.model.predict(x)
        idx = np.argmax(preds)
        return CLASS_LABELS[idx], preds