import numpy as np
import cv2
import tflite_runtime.interpreter as tfi

# Charger le modèle TFLite
interpreter = tfi.Interpreter(model_path="../code/best_full_integer_quant_edgetpu.tflite")
interpreter.allocate_tensors()

# Prétraiter l'image
image = cv2.imread("../assets/balle_dessus_filet .png")
image = cv2.resize(image, (224, 224))  # Adapter la taille à celle du modèle
image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0  # Normaliser

# Exécution de l'inférence
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], image)
interpreter.invoke()

# Afficher le résultat
output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
print("Sortie : ", output)
