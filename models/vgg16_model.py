from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from models.pipeline import BaseVoiceClassifier

class VGG16Classifier(BaseVoiceClassifier):
    def __init__(self, dataset_path):
        super().__init__(dataset_path, image_size=(224, 224), preprocess_fn=preprocess_input, epochs=10)

    def build_model(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers[:-4]:
            layer.trainable = False

        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.train_generator.class_indices), activation='softmax')
        ])