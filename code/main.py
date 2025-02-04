import cv2
import os
import glob
import json
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects

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

# Initialisation du modèle TFLite avec TPU Coral
model_path = "best_full_integer_quant_edgetpu.tflite"  # Modifiez le chemin si nécessaire
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

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
output_json_path = os.path.join(output_dir, "detections.json")
detections_data = {
    "video": video_path,
    "fps": fps,
    "total_frames": total_frames,
    "detections": []
}

# Lecture frame par frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fin de la vidéo ou erreur lors de la lecture.")
        break

    print(f"Traitement de la frame {frame_counter}/{total_frames}...")

    # Prétraitement de l'image
    input_shape = input_size(interpreter)
    resized_frame = cv2.resize(frame, input_shape)
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    resized_frame = np.expand_dims(resized_frame, axis=0)

    # Détection avec TFLite et TPU Coral
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], resized_frame)
    interpreter.invoke()
    objs = get_objects(interpreter, 0.4)  # Confiance minimale de 0.4

    # Vérifier si des objets sont détectés
    if objs:
        print(f"Balles détectées dans la frame {frame_counter} : {len(objs)}")

        detection_info = {
            "frame": frame_counter,
            "detections": []
        }

        for obj in objs:
            bbox = obj.bbox
            x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax

            # Ajouter les coordonnées et la confiance au JSON
            detection_info["detections"].append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": obj.score
            })

            # Dessiner le rectangle autour de la balle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Balle ({obj.score:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Ajouter les détections de la frame au fichier JSON
        detections_data["detections"].append(detection_info)

        # Sauvegarder l'image annotée dans le dossier /assets/
        output_image_path = os.path.join(output_dir, f"detection_{frame_counter}.png")
        cv2.imwrite(output_image_path, frame)
    else:
        print(f"Aucune balle détectée dans la frame {frame_counter}.")

    # Incrémenter le compteur de frames
    frame_counter += 1

# Sauvegarder les données de détection dans un fichier JSON
with open(output_json_path, "w") as json_file:
    json.dump(detections_data, json_file, indent=4)
print(f"Données de détection sauvegardées dans : {output_json_path}")

# Libérer les ressources
cap.release()
print("Traitement terminé.")