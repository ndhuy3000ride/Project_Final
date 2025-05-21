import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

class BaseImageTrainer:
    def __init__(self, dataset_path, model_save_path, image_size, batch_size, epochs, datagen_args=None):
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.datagen_args = datagen_args or {}

        self.train_generator = None
        self.validation_generator = None
        self.model = None
        self.history = None
        self.class_indices = None

    def prepare_data(self):
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            **self.datagen_args
        )
        self.train_generator = datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        self.validation_generator = datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        self.class_indices = self.train_generator.class_indices
        # Save class labels
        with open(os.path.join(os.path.dirname(self.model_save_path), 'class_labels.json'), 'w') as f:
            json.dump(self.class_indices, f)

    def build_model(self):
        raise NotImplementedError("Override build_model in child class!")

    def compile_model(self, optimizer, loss='categorical_crossentropy'):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        self.model.summary()

    def get_callbacks(self):
        return [
            ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1
            ),
            ModelCheckpoint(
                self.model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1
            )
        ]

    def train(self):
        self.prepare_data()
        self.build_model()
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.validation_generator,
            epochs=self.epochs,
            callbacks=self.get_callbacks()
        )
        self.model.save(self.model_save_path.replace('.h5', '_final.h5'))
