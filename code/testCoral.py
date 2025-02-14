import argparse
import cv2
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
import numpy as np
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', help='File path of Tflite model.', required=True)
    parser.add_argument('--label', help='File path of label file.', required=True)
    parser.add_argument(
        '--image', help='File path of the image to be recognized.', required=True)
    parser.add_argument(
        '--top_k', type=int, default=3, help='Number of top results to display.')
    parser.add_argument(
        '--threshold', type=float, default=0.4, help='Detection threshold.')
    args = parser.parse_args()

    # Prepare labels.
    labels = dataset_utils.read_label_file(args.label)
    # Initialize engine.
    engine = DetectionEngine(args.model)
    # Run inference.
    img = Image.open(args.image)
    img_np = np.array(img)
    results = engine.detect_with_image(img, threshold=args.threshold, top_k=args.top_k, keep_aspect_ratio=True,
                                       relative_coord=False)
    if results:
        for result in results:
            print('---------------------------')
            label_id = result.label_id
            if label_id in labels:
                print(labels[label_id])
            else:
                print(f"Label ID {label_id} not found in labels.")
            print('Score : ', result.score)
            box = result.bounding_box.flatten().tolist()
            print('Bounding box:', box)
            # Draw rectangle around detected object
            cv2.rectangle(img_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        # Save the annotated image
        annotated_image = Image.fromarray(img_np)
        annotated_image = annotated_image.convert("RGB")  # Convert to RGB mode
        os.makedirs('assets', exist_ok=True)  # Create the assets directory if it doesn't exist
        save_path = os.path.join('assets', 'annotated_image.jpg')
        annotated_image.save(save_path)
        print(f"Annotated image saved as '{save_path}'")

        # Open the annotated image using fim
        subprocess.run(['fim', save_path])
    else:
        print("No results found.")


if __name__ == '__main__':
    main()