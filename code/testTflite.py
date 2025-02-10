import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Définir le chemin de l'image annotée
output_path = "photo_annotated.png"
if os.path.exists(output_path):
    os.remove(output_path)
    print(f"Fichier existant '{output_path}' supprimé.")

# Chemin du modèle TFLite
model_path = "../code/best_full_integer_quant_edgetpu.tflite"  # Remplace par le chemin vers ton modèle TFLite

# Charger le modèle TFLite
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Charger l'image
image_path = "../assets/balle_dessus_filet .png"  # Remplace par le chemin de ton image
image = cv2.imread(image_path)
assert image is not None, "Erreur : Impossible de charger l'image."

# Prétraiter l'image pour le modèle (mettre à la bonne taille et normaliser les pixels)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Assurer que l'image a la bonne taille pour le modèle
input_shape = input_details[0]['shape']
input_image = cv2.resize(image, (input_shape[2], input_shape[1]))  # Ajuster la taille
input_image = np.expand_dims(input_image, axis=0)  # Ajouter une dimension pour le batch
input_image = np.float32(input_image)  # Assurer le bon type de données

# Effectuer une prédiction
interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()

# Récupérer les résultats de la détection
boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Boîtes de détection
class_ids = interpreter.get_tensor(output_details[1]['index'])[0]  # Identifiants des classes
confidences = interpreter.get_tensor(output_details[2]['index'])[0]  # Confiance des prédictions

# Filtrer les prédictions en fonction de la confiance
for i in range(len(boxes)):
    if confidences[i] > 0.4:  # Seuil de confiance
        box = boxes[i]
        class_id = int(class_ids[i])
        confidence = confidences[i]

        # Calcul des coins du rectangle
        y1, x1, y2, x2 = box
        x1, y1, x2, y2 = int(x1 * image.shape[1]), int(y1 * image.shape[0]), int(x2 * image.shape[1]), int(y2 * image.shape[0])

        # Dessiner le rectangle autour de l'objet détecté
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"Class {class_id} ({confidence:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

# Convertir l'image BGR en RGB pour un affichage correct avec Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Sauvegarder l'image annotée
cv2.imwrite(output_path, image)
print(f"Image annotée sauvegardée sous : {output_path}")

# Afficher l'image annotée avec Matplotlib
plt.imshow(image_rgb)
plt.axis("off")
plt.title("Résultat")
plt.show()
