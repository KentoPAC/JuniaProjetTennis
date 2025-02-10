import cv2
from tflite_runtime.interpreter import load_delegate, Interpreter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time


def scale_image(frame, new_size=(224, 224)):
    height, width, _ = frame.shape  # Image shape
    new_width, new_height = new_size  # Target shape
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return frame[top:bottom, left:right, :]


def time_elapsed(start_time, event):
    duration = round((time.time() - start_time) * 1000, 2)
    print(f">>> {duration} ms ({event})")


# Chargement du modèle et des labels
model_path = "mobilenet_v1_1.0_224_quant_edgetpu.tflite"
label_path = "labels_mobilenet_quant_v1_224.txt"

top_k_results = 2

with open(label_path, 'r') as f:
    labels = list(map(str.strip, f.readlines()))

interpreter = Interpreter(model_path=model_path,
                          experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
size = input_shape[:2] if len(input_shape) == 3 else input_shape[1:3]

threshold = 0.5

plt.ion()
plt.tight_layout()
fig = plt.gcf()
fig.canvas.set_window_title('TensorFlow Lite')
fig.suptitle('Image Classification')
ax = plt.gca()
ax.set_axis_off()
tmp = np.zeros([480, 640, 3], np.uint8)
preview = ax.imshow(tmp)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra")
    exit()

while True:
    start_time = time.time()

    # Capture d'image
    start_t1 = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la capture de l'image")
        break
    img = scale_image(frame)
    time_elapsed(start_t1, "camera capture")

    # Prétraitement et exécution de l'inférence
    start_t2 = time.time()
    input_data = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    top_k_indices = np.argsort(predictions)[::-1][:top_k_results]

    pred_max = predictions[top_k_indices[0]] / 255.0
    lbl_max = labels[top_k_indices[0]]

    text_display = "___" if pred_max < threshold else f"{lbl_max} ({round(pred_max * 100)}%)"
    time_elapsed(start_t2, "inference")

    # Affichage des résultats
    start_t3 = time.time()
    cv2.putText(frame, text_display, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera View', frame)
    time_elapsed(start_t3, "preview")

    print(lbl_max, pred_max)
    print("********************************")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
