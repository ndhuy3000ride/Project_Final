import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class BaseVoiceClassifier:
    def __init__(self, dataset_path, image_size=(224, 224), batch_size=32, epochs=20, preprocess_fn=None):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.preprocess_fn = preprocess_fn
        self.train_generator = None
        self.validation_generator = None
        self.model = None

    def prepare_data(self):
        datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_fn,
            validation_split=0.2,
            zoom_range=0.05,
            width_shift_range=0.05,
            height_shift_range=0.05
        )

        self.train_generator = datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

        self.validation_generator = datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )

        # Save class labels
        with open(f'class_labels_{self.__class__.__name__.lower()}.json', 'w') as f:
            json.dump(self.train_generator.class_indices, f)

    def compile_and_train(self, model_save_path):
        self.prepare_data()
        self.build_model()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        history = self.model.fit(
            self.train_generator,
            validation_data=self.validation_generator,
            epochs=self.epochs
        )

        self.model.save(model_save_path)

        loss, acc = self.model.evaluate(self.validation_generator)
        print(f"✅ Validation Accuracy ({self.__class__.__name__}): {acc * 100:.2f}%")