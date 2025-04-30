import cv2
from ultralytics import YOLO
import os
import time
import json
import glob

def ball(output_dir, video_path, model_path):
    output_jsonl = os.path.join(output_dir, "balle.jsonl")


    # Supprimer l'ancien fichier JSONL s'il existe
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)

    # Création du dossier de sortie s’il n’existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Nettoyage des anciennes images
    for file in glob.glob(os.path.join(output_dir, "detection_*.png")):
        os.remove(file)

    # Chargement du modèle
    model = YOLO(model_path)

    # Ouverture de la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_counter = 0
    start_time = time.time()

    with open(output_jsonl, "a") as f_json:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_start_time = time.time()

            # Détection avec confiance normale
            results = model.predict(frame, conf=0.15, iou=0.15, verbose=False)
            boxes = results[0].boxes

            # Retenter si aucune box trouvée
            if len(boxes) == 0:
                results = model.predict(frame, conf=0.12, iou=0.15, verbose=False)
                boxes = results[0].boxes

            best_box = None
            best_conf = 0
            for box in boxes:
                conf = box.conf.item()
                if conf > best_conf:
                    best_conf = conf
                    best_box = box

            detection_info = {
                "frame": frame_counter,
                "fps": fps,
                "total_frames": total_frames,
                "detections": [],
                "no_detection": False
            }

            if best_box:
                Ball_X, Ball_Y = (
                    best_box.xywh[0][0].item(),
                    best_box.xywh[0][1].item(),
                )

                # Annoter uniquement la position détectée
                cv2.circle(frame, (int(Ball_X), int(Ball_Y)), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"Balle ({best_conf:.2f})",
                    (int(Ball_X), int(Ball_Y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                frame_processing_time = time.time() - frame_start_time

                detection_info["detections"].append({
                    "Ball_X": int(Ball_X),
                    "Ball_Y": int(Ball_Y),
                    "confidence": best_conf,
                    "temps_detection": frame_processing_time
                })

                output_image_path = os.path.join(output_dir, f"detection_{frame_counter}.png")
                cv2.imwrite(output_image_path, frame)

            else:
                detection_info["no_detection"] = True

            # Écriture incrémentale ligne par ligne
            f_json.write(json.dumps(detection_info) + "\n")

            frame_counter += 1

    cap.release()
    total_processing_time = time.time() - start_time
    print(f"Traitement terminé en {total_processing_time:.2f} secondes.")
