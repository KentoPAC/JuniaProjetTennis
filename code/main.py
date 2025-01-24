import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Initialisation du modèle YOLO
model_path = "../code/best.pt"
model = YOLO(model_path)

# Charger la vidéo
video_path = "../Vidéos/MatchV1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

# Dossier de sauvegarde
output_dir = "../assets/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Crée le dossier s'il n'existe pas

frame_counter = 0  # Compteur pour différencier les images sauvegardées

# Lire les propriétés de la vidéo
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Vidéo chargée : {total_frames} frames à {fps:.2f} FPS.")

# Lecture frame par frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fin de la vidéo ou erreur lors de la lecture.")
        break

    print(f"Traitement de la frame {frame_counter}/{total_frames}...")

    # Détection avec YOLO
    results = model.predict(frame, conf=0.4, iou=0.3, verbose=False)

    # Vérifier si des objets sont détectés
    if results and results[0].boxes:
        print(f"Balles détectées dans la frame {frame_counter} : {len(results[0].boxes)}")
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


        # Sauvegarder l'image annotée dans le dossier /assets/
        output_image_path = os.path.join(output_dir, f"detection_{frame_counter}.png")
        cv2.imwrite(output_image_path, frame)
        print(f"Image détectée et sauvegardée sous : {output_image_path}")
    else:
        print(f"Aucune balle détectée dans la frame {frame_counter}.")

    # Incrémenter le compteur de frames
    frame_counter += 1

# Libérer les ressources
cap.release()
print("Traitement terminé.")
