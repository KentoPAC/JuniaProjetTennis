import cv2
from ultralytics import YOLO
import os
import time
import json
import glob

model_path = "best2.pt"
video_path = "../../Vidéos/Test2.mp4"
output_dir = "../../assets/"
output_json = os.path.join(output_dir, "balle.json")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file in glob.glob(os.path.join(output_dir, "detection_*.png")):
    os.remove(file)

model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Vidéo chargée : {total_frames} frames à {fps:.2f} FPS.")

# ---------- Données JSON ----------
detections_data = {
    "video": video_path,
    "fps": fps,
    "total_frames": total_frames,
    "duree_total": 0,
    "detections": []
}

frame_counter = 0
start_time = time.time()

trajectory_points = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_start_time = time.time()

    # Première tentative avec confiance normale
    results = model.predict(frame, conf=0.15, iou=0.15, verbose=False)
    boxes = results[0].boxes

    # Si aucune détection : retente avec confiance faible
    if len(boxes) == 0:
        results = model.predict(frame, conf=0.12, iou=0.15, verbose=False)
        boxes = results[0].boxes

    # Prendre la meilleure box (la plus confiante)
    best_box = None
    best_conf = 0
    for box in boxes:
        conf = box.conf.item()
        if conf > best_conf:
            best_conf = conf
            best_box = box

    detection_info = {
        "frame": frame_counter,
        "detections": [],
        "no_detection": False
    }

    if best_box:
        Ball_X, Ball_Y = (
            best_box.xywh[0][0].item(),
            best_box.xywh[0][1].item(),
        )

        print(f"  Balle détectée : x={Ball_X:.2f}, y={Ball_Y:.2f}")

        # Sauvegarde du point pour la trajectoire
        trajectory_points.append((int(Ball_X), int(Ball_Y)))

        # Dessiner la trajectoire (lignes entre points successifs)
        for i in range(1, len(trajectory_points)):
            cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 255, 0), 2)

        # Annoter le point courant
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

        # Sauvegarde image
        output_image_path = os.path.join(output_dir, f"detection_{frame_counter}.png")
        cv2.imwrite(output_image_path, frame)

    else:
        detection_info["no_detection"] = True

    detections_data["detections"].append(detection_info)
    frame_counter += 1

cap.release()
total_processing_time = time.time() - start_time
detections_data["duree_total"] = total_processing_time

with open(output_json, "w") as f:
    json.dump(detections_data, f, indent=4)

print(f"\nTraitement terminé en {total_processing_time:.2f} secondes.")
print(f"Fichier JSON enregistré : {output_json}")
