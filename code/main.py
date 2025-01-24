import cv2
from ultralytics import YOLO
import os
import json

# Initialisation de l'API Roboflow
model_path = "../code/best.pt"  # Modifiez le chemin si nécessaire
model = YOLO(model_path)

# Liste des chemins des images à traiter (exemple avec 10 images)
image_paths = [
    f"../images/image_{i}.png" for i in range(1, 14)
]

# Structure pour stocker les résultats
results_json = {}

# Traiter chaque image
for image_path in image_paths:
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : Impossible de charger l'image {image_path}.")
        continue

    # Analyse de l'image
    results = model.predict(image, conf=0.3, iou=0.2)

    # Vérifier si des prédictions existent
    image_results = []
    if results and results[0].boxes:
        for box in results[0].boxes:
            # Récupérer les coordonnées du rectangle de détection
            x, y, width, height = (
                box.xywh[0][0].item(),
                box.xywh[0][1].item(),
                box.xywh[0][2].item(),
                box.xywh[0][3].item(),
            )
            conf = box.conf.item()
            label = int(box.cls.item())

            # Ajouter les données à la liste des résultats de l'image
            image_results.append({
                "label": label,
                "confidence": conf,
                "coordinates": {
                    "x_center": x,
                    "y_center": y,
                    "width": width,
                    "height": height
                }
            })

    # Sauvegarder les résultats pour cette image
    results_json[os.path.basename(image_path)] = image_results

# Sauvegarder tous les résultats dans un fichier JSON
output_json_path = "detection_results.json"
with open(output_json_path, "w") as json_file:
    json.dump(results_json, json_file, indent=4)

print(f"Résultats enregistrés dans le fichier : {output_json_path}")
