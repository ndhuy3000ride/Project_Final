from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import Bidirectional, LSTM
from .base_trainer import BaseImageTrainer
import tensorflow as tf

class CNNLSTMTrainer(BaseImageTrainer):
    def build_model(self):
        num_classes = len(self.class_indices)
        inputs = layers.Input(shape=(self.image_size[0], self.image_size[1], 3))
        x = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.SpatialDropout2D(0.2)(x)

        x = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.SpatialDropout2D(0.2)(x)

        x = layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.SpatialDropout2D(0.2)(x)

        shape_before_reshape = tf.keras.backend.int_shape(x)
        x = layers.Reshape((shape_before_reshape[1] * shape_before_reshape[2], shape_before_reshape[3]))(x)
        x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))(x)
        x = Bidirectional(LSTM(128, dropout=0.3))(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        self.model = models.Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, weight_decay=1e-5)
        self.compile_model(optimizer)
