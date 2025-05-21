from tensorflow.keras import layers, models, regularizers
from .base_trainer import BaseImageTrainer
import tensorflow as tf

class CustomCNNTrainer(BaseImageTrainer):
    def build_model(self):
        num_classes = len(self.class_indices)
        self.model = models.Sequential([
            layers.Conv2D(64, (3,3), activation='relu', input_shape=(self.image_size[0], self.image_size[1], 3), kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.SpatialDropout2D(0.2),
            layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.SpatialDropout2D(0.2),
            layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.SpatialDropout2D(0.2),
            layers.Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.GlobalMaxPooling2D(),
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, weight_decay=1e-5)
        self.compile_model(optimizer)
