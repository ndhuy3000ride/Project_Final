import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Input
from tensorflow.keras.applications import VGG16
from .base_trainer import BaseImageTrainer

def create_custom_cnn_branch(input_tensor):
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.SpatialDropout2D(0.2)(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.SpatialDropout2D(0.2)(x)
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.SpatialDropout2D(0.2)(x)
    x = layers.Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.BatchNormalization()(x)
    return x

class VGG16CNNFusionTrainer(BaseImageTrainer):
    def build_model(self):
        n_classes = len(self.class_indices)
        input_tensor = Input(shape=self.image_size + (3,))

        # VGG16 branch
        vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=self.image_size + (3,))
        vgg_base.trainable = False  # Freeze toàn bộ VGG16
        vgg_feat = layers.GlobalAveragePooling2D()(vgg_base(input_tensor))
        vgg_feat = layers.BatchNormalization()(vgg_feat)

        # Custom CNN branch
        cnn_feat = create_custom_cnn_branch(input_tensor)

        # Fusion
        merged = layers.Concatenate()([vgg_feat, cnn_feat])

        # Head classifier
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(merged)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(n_classes, activation='softmax')(x)

        self.model = Model(inputs=input_tensor, outputs=output)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, weight_decay=1e-5)
        self.compile_model(optimizer)

    def fine_tune(self, train_generator=None, validation_generator=None, epochs=10):
        # Unfreeze block5 of VGG16 only
        vgg_base = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model) and 'vgg16' in layer.name.lower():
                vgg_base = layer
                break

        if vgg_base is not None:
            set_trainable = False
            for layer in vgg_base.layers:
                if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
                    set_trainable = True
                if set_trainable:
                    layer.trainable = True
            for layer in vgg_base.layers[-8:]:
                print(layer.name, layer.trainable)
        else:
            print("Warning: vgg_base not found for fine-tuning!")

        # Compile lại với learning rate nhỏ + label smoothing
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=['accuracy']
        )

        # Fine-tune
        print("Fine-tuning last VGG16 block...")
        history_ft = self.model.fit(
            train_generator or self.train_generator,
            validation_data=validation_generator or self.validation_generator,
            epochs=epochs,
            callbacks=self.get_callbacks()
        )
        return history_ft
