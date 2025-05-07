import argparse
from cnn_model import CNNClassifier
from resnet50_model import ResNet50Classifier
from vgg16_model import VGG16Classifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "resnet", "vgg"], help="Model to train")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output", type=str, default="model.h5", help="Path to save model")
    args = parser.parse_args()

    if args.model == "cnn":
        model = CNNClassifier(dataset_path=args.data)
    elif args.model == "resnet":
        model = ResNet50Classifier(dataset_path=args.data)
    elif args.model == "vgg":
        model = VGG16Classifier(dataset_path=args.data)

    model.compile_and_train(model_save_path=args.output)
