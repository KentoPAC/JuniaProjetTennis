import cv2
from ultralytics import YOLO
import os
import glob
import json
import time  # Importer le module time

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

# Initialisation du modèle YOLO
model_path = "../code/best.pt"
model = YOLO(model_path)

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

    # Détection avec YOLO
    results = model.predict(frame, conf=0.3, iou=0.3, verbose=False)

    # Vérifier si des objets sont détectés
    if results and results[0].boxes:
        print(f"Balles détectées dans la frame {frame_counter} : {len(results[0].boxes)}")

        detection_info = {
            "frame": frame_counter,
            "detections": []
        }

        for box in results[0].boxes:
            # Récupérer les coordonnées du rectangle de détection
            x, y, width, height = (
                box.xywh[0][0].item(),
                box.xywh[0][1].item(),
                box.xywh[0][2].item(),
                box.xywh[0][3].item(),
            )
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)

            # Ajouter les coordonnées et la confiance au JSON
            detection_info["detections"].append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": box.conf.item(),
                "temps_detection": 0  # Placeholder pour le temps de détection
            })

            # Dessiner le rectangle autour de la balle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Balle ({box.conf.item():.2f})",
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
    else:
        print(f"Aucune balle détectée dans la frame {frame_counter}.")

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