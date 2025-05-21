import tensorflow as tf
from tensorflow.keras import layers, models
from .base_trainer import BaseImageTrainer
from tensorflow import keras

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class ViTTrainer(BaseImageTrainer):
    def build_model(self, patch_size=16, projection_dim=64, num_heads=4,
                    transformer_units=[128, 64], transformer_layers=8,
                    mlp_head_units=[256, 128]):
        num_classes = len(self.class_indices)
        input_shape = self.image_size + (3,)
        inputs = layers.Input(shape=input_shape)
        num_patches = (self.image_size[0] // patch_size) * (self.image_size[1] // patch_size)
        patches = Patches(patch_size)(inputs)
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        for _ in range(transformer_layers):
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim // num_heads, dropout=0.1
            )(x1, x1)
            x2 = layers.Add()([attention_output, encoded_patches])
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            encoded_patches = layers.Add()([x3, x2])

        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.GlobalAveragePooling1D()(representation)
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3)
        logits = layers.Dense(num_classes, activation="softmax")(features)

        self.model = keras.Model(inputs=inputs, outputs=logits)
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.compile_model(optimizer, loss=keras.losses.CategoricalCrossentropy())
