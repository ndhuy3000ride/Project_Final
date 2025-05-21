from models.cnn_trainer import CustomCNNTrainer
from models.vgg16_trainer import VGG16Trainer
from models.resnet50_trainer import ResNet50Trainer
from models.mobilenetv2_trainer import MobileNetV2Trainer
from models.cnn_lstm_trainer import CNNLSTMTrainer
from models.vit_trainer import ViTTrainer
from models.vgg16_cnn_fusion_trainer import VGG16CNNFusionTrainer


DATASET_PATH = "data/path"

if __name__ == "__main__":
    # VD: train VGG16
    vgg_trainer = VGG16Trainer(
        dataset_path=DATASET_PATH,
        model_save_path="model/path",
        image_size=(224, 224),
        batch_size=16,
        epochs=30,
        datagen_args={
            "rotation_range": 10, "width_shift_range": 0.05, "height_shift_range": 0.05,
            "zoom_range": 0.1, "brightness_range": [0.9, 1.1],
            "horizontal_flip": False, "fill_mode": "constant", "cval": 0
        }
    )

    # VD: train MobileNetV2
    mobilenet_trainer = MobileNetV2Trainer(
        dataset_path=DATASET_PATH,
        model_save_path="model/path",
        image_size=(128, 128),
        batch_size=32,
        epochs=30,
        datagen_args={
            "rotation_range": 10, "width_shift_range": 0.05, "height_shift_range": 0.05,
            "zoom_range": 0.1, "brightness_range": [0.9, 1.1],
            "horizontal_flip": False, "fill_mode": "constant", "cval": 0
        }
    )

    # Tương tự cho ResNet50, ViT, CNN-LSTM...
    custom_cnn_trainer = CustomCNNTrainer(
        dataset_path=DATASET_PATH,
        model_save_path="model/path",
        image_size=(128, 128),
        batch_size=32,
        epochs=30,
        datagen_args={
            "rotation_range": 10, "width_shift_range": 0.05, "height_shift_range": 0.05,
            "zoom_range": 0.1, "brightness_range": [0.9, 1.1],
            "horizontal_flip": False, "fill_mode": "constant", "cval": 0
        }
    )

    resnet_50_trainer = ResNet50Trainer(
        dataset_path=DATASET_PATH,
        model_save_path="model/path",
        image_size=(224, 224),
        batch_size=32,
        epochs=30,
        datagen_args={
            "rotation_range": 10, "width_shift_range": 0.05, "height_shift_range": 0.05,
            "zoom_range": 0.1, "brightness_range": [0.9, 1.1],
            "horizontal_flip": False, "fill_mode": "constant", "cval": 0
        }
    )

    cnn_lstm_trainer = CNNLSTMTrainer(
        dataset_path=DATASET_PATH,
        model_save_path="model/path",
        image_size=(128, 128),
        batch_size=32,
        epochs=30,
        datagen_args={
            "rotation_range": 10, "width_shift_range": 0.05, "height_shift_range": 0.05,
            "zoom_range": 0.1, "brightness_range": [0.9, 1.1],
            "horizontal_flip": False, "fill_mode": "constant", "cval": 0
        }
    )

    vit_trainer = ViTTrainer(
        dataset_path=DATASET_PATH,
        model_save_path="model/path",
        image_size=(128, 128),
        batch_size=32,
        epochs=60,
        datagen_args={
            "rotation_range": 10, "width_shift_range": 0.05, "height_shift_range": 0.05,
            "zoom_range": 0.1, "brightness_range": [0.9, 1.1],
            "horizontal_flip": False, "fill_mode": "constant", "cval": 0
        }
    )
    
    vgg16_cnn_fusion_trainer = VGG16CNNFusionTrainer(
    dataset_path="DATASET_PATH",
    model_save_path="model/path",
    image_size=(128, 128),
    batch_size=32,
    epochs=30,
    datagen_args={
        "rotation_range": 15,
        "width_shift_range": 0.05,
        "height_shift_range": 0.05,
        "zoom_range": 0.05,
        "brightness_range": [0.9, 1.1],
        "horizontal_flip": True,
        "fill_mode": "constant",
        "cval": 0
    }
)

#Chon
vgg_trainer.train()
custom_cnn_trainer.train()
resnet_50_trainer.train()
mobilenet_trainer.train()
vit_trainer.train()
cnn_lstm_trainer.train()

vgg16_cnn_fusion_trainer.train()
vgg16_cnn_fusion_trainer.fine_tune(epochs = 10)
