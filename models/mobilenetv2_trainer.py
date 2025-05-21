from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, regularizers
from .base_trainer import BaseImageTrainer
import tensorflow as tf

class MobileNetV2Trainer(BaseImageTrainer):
    def build_model(self):
        num_classes = len(self.class_indices)
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(self.image_size[0], self.image_size[1], 3))
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, weight_decay=1e-5)
        self.compile_model(optimizer)
