import cv2
import os
import glob
import json
import time
import numpy as np
import tensorflow as tf

# Chemin du dossier de sauvegarde
output_dir = "../assets/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Crée le dossier s'il n'existe pas

# Supprimer les fichiers existants correspondant au motif "detection_*.png"
files_to_delete = glob.glob(os.path.join(output_dir, "detection_*.png"))
for file in files_to_delete:
    try:
        os.remove(file)
    except Exception:
        pass  # Supprime les messages d'erreur pour une console simplifiée

# Supprimer le fichier detections.json s'il existe
output_json_path = os.path.join(output_dir, "detections.json")
if os.path.exists(output_json_path):
    os.remove(output_json_path)

# Chemin du modèle TFLite
model_path = "../code/best_full_integer_quant_edgetpu.tflite"  # Remplace par le chemin vers ton modèle TFLite

# Charger le modèle TFLite
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Charger le flux vidéo de la caméra
cap = cv2.VideoCapture(0)  # Utiliser l'index 0 pour la première caméra connectée

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

frame_counter = 0  # Compteur pour différencier les images sauvegardées

# Lire les propriétés de la caméra
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Flux vidéo en direct à {fps:.2f} FPS.")

# Fichier JSON pour enregistrer les détections
detections_data = {
    "video": "Flux vidéo en direct",
    "fps": fps,
    "total_frames": 0,  # Le nombre total de frames n'est pas connu à l'avance pour un flux en direct
    "duree_total": 0,  # Temps total de traitement
    "detections": []
}

# Démarrer le timer pour le temps total de traitement
start_time = time.time()

# Lecture frame par frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la lecture du flux vidéo.")
        break

    print(f"Traitement de la frame {frame_counter}...")

    # Démarrer le timer pour le temps de traitement de la frame
    frame_start_time = time.time()

    # Prétraiter l'image pour le modèle (mettre à la bonne taille et normaliser les pixels)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Assurer que l'image a la bonne taille pour le modèle
    input_shape = input_details[0]['shape']
    input_image = cv2.resize(frame, (input_shape[2], input_shape[1]))  # Ajuster la taille
    input_image = np.expand_dims(input_image, axis=0)  # Ajouter une dimension pour le batch
    input_image = np.float32(input_image)  # Assurer le bon type de données

    # Effectuer une prédiction
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    # Récupérer les résultats de la détection
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Boîtes de détection
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]  # Identifiants des classes
    confidences = interpreter.get_tensor(output_details[2]['index'])[0]  # Confiance des prédictions

    detection_info = {
        "frame": frame_counter,
        "detections": []
    }

    # Filtrer les prédictions en fonction de la confiance
    for i in range(len(boxes)):
        if confidences[i] > 0.4:  # Seuil de confiance
            box = boxes[i]
            class_id = int(class_ids[i])
            confidence = confidences[i]

            # Calcul des coins du rectangle
            y1, x1, y2, x2 = box
            x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])

            # Ajouter les coordonnées et la confiance au JSON
            detection_info["detections"].append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence,
                "temps_detection": 0  # Placeholder pour le temps de détection
            })

            # Dessiner le rectangle autour de l'objet détecté
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Class {class_id} ({confidence:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # Calculer le temps de traitement de la frame
    frame_end_time = time.time()
    frame_processing_time = frame_end_time - frame_start_time

    # Ajouter le temps de détection au JSON
    for detection in detection_info["detections"]:
        detection["temps_detection"] = frame_processing_time

    # Ajouter les détections de la frame au fichier JSON
    detections_data["detections"].append(detection_info)

    # Sauvegarder l'image annotée dans le dossier /assets/
    output_image_path = os.path.join(output_dir, f"detection_{frame_counter}.png")
    cv2.imwrite(output_image_path, frame)

    # Incrémenter le compteur de frames
    frame_counter += 1

# Calculer le temps total de traitement
end_time = time.time()
total_processing_time = end_time - start_time
detections_data["duree_total"] = total_processing_time

# Sauvegarder les données de détection dans un fichier JSON
with open(output_json_path, "w") as json_file:
    json.dump(detections_data, json_file, indent=4)
print(f"Données de détection sauvegardées dans : {output_json_path}")

# Libérer les ressources
cap.release()
print("Traitement terminé.")