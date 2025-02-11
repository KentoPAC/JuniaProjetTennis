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
    parser.add_argument(
        '--top_k', type=int, default=3, help='Number of top results to display.')
    args = parser.parse_args()

    # Prepare labels.
    labels = dataset_utils.read_label_file(args.label)
    # Initialize engine.
    engine = ClassificationEngine(args.model)
    # Run inference.
    img = Image.open(args.image)
    results = engine.classify_with_image(img, top_k=args.top_k)
    if results:
        for result in results:
            print('---------------------------')
            label_id = result[0]
            if label_id in labels:
                print(labels[label_id])
            else:
                print(f"Label ID {label_id} not found in labels.")
            print('Score : ', result[1])
    else:
        print("No results found.")

if __name__ == '__main__':
    main()