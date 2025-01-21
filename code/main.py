import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

output_path = "photo_annotated.png"
if os.path.exists(output_path):
  os.remove(output_path)
  print(f"Fichier existant '{output_path}' supprimé.")

# Initialisation de l'API Roboflow
model_path = "../code/best.pt"
model = YOLO(model_path)

# Charger l'image
image_path = "../assets/balle_dessus_filet .png"  # Chemin de l'image
image = cv2.imread(image_path)
assert image is not None, "Erreur : Impossible de charger l'image."

# Envoyer l'image pour analyse
print("Analyse de l'image en cours...")
results = model.predict(image_path, conf=0.4, iou=0.3)

# Vérifier si des prédictions existent
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
        label = box.cls.item()

        # Calcul des coins du rectangle
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)

        # Dessiner le rectangle autour de la balle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label} ({conf:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
else:
    print("Aucune balle détectée dans l'image.")

# Convertir l'image BGR en RGB pour un affichage correct avec Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Sauvegarder l'image annotée
output_path = "photo_annotated.png"
cv2.imwrite(output_path, image)
print(f"Image annotée sauvegardée sous : {output_path}")

# Afficher l'image annotée avec Matplotlib
plt.imshow(image_rgb)
plt.axis("off")
plt.title("Résultat")
plt.show()

