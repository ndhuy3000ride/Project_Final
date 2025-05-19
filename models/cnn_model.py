from tensorflow.keras import layers, models
from models.pipeline import BaseVoiceClassifier

class CNNClassifier(BaseVoiceClassifier):
    def __init__(self, dataset_path):
        super().__init__(dataset_path, image_size=(128, 128), epochs=20, preprocess_fn=lambda x: x / 255.0)

    def build_model(self):
        self.model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(self.image_size[0], self.image_size[1], 3)),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(128, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.train_generator.class_indices), activation='softmax')
        ])