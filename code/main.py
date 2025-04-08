import cv2
from ultralytics import YOLO
import os
import glob
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

# Initialisation du modèle YOLO
model_path = "../code/best.pt"
model = YOLO(model_path)

# Charger la vidéo
video_path = "../Vidéos/VideoBallV2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

frame_counter = 0  # Compteur pour différencier les images sauvegardées

# Lire les propriétés de la vidéo
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Vidéo chargée : {total_frames} frames à {fps:.2f} FPS.")

# Fichier JSON pour enregistrer les détections
detections_data = {
    "video": video_path,
    "fps": fps,
    "total_frames": total_frames,
    "duree_total": 0,  # Temps total de traitement
    "detections": []
}

# Démarrer le timer pour le temps total de traitement
start_time = time.time()

# Lecture frame par frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fin de la vidéo ou erreur lors de la lecture.")
        break

    print(f"Traitement de la frame {frame_counter}/{total_frames}...")

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

            # Affichage en console
            print(f"  -> Balle détectée : x={x}, y={y} confiance={box.conf.item():.2f}")

            # Ajouter les coordonnées et la confiance au JSON
            detection_info["detections"].append({
                "x": int(x),
                "y": int(y),
                "confidence": box.conf.item(),
                "temps_detection": 0
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

# Libérer les ressources
cap.release()
print("Traitement terminé.")
