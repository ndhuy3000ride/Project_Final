from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from base_model import BaseVoiceClassifier

class ResNet50Classifier(BaseVoiceClassifier):
    def __init__(self, dataset_path):
        super().__init__(dataset_path, image_size=(224, 224), preprocess_fn=preprocess_input)

    def build_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.train_generator.class_indices), activation='softmax')
        ])