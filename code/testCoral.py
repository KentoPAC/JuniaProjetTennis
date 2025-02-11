import argparse
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.utils import dataset_utils
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', help='File path of Tflite model.', required=True)
    parser.add_argument('--label', help='File path of label file.', required=True)
    parser.add_argument(
        '--image', help='File path of the image to be recognized.', required=True)
    args = parser.parse_args()

    # Prepare labels.
    labels = dataset_utils.read_label_file(args.label)
    # Initialize engine.
    engine = ClassificationEngine(args.model)
    # Run inference.
    img = Image.open(args.image)
    for result in engine.classify_with_image(img, top_k=3):
        print('---------------------------')
        label_id = result[0]
        if label_id in labels:
            if label_id == 853:  # ID 853 corresponds to "tennis ball" in your labels file
                print(labels[label_id])
                print('Score : ', result[1])
        else:
            print(f"Label ID {label_id} not found in labels.")

if __name__ == '__main__':
    main()