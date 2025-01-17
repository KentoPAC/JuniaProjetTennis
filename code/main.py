import cv2
from roboflow import Roboflow

# Initialisation de l'API Roboflow
rf = Roboflow(api_key="HGDPPCrv2JXA19Dfds89")
project = rf.workspace().project("tennis_ball_yolov8-4qhxa-qjgrh")
model = project.version("1").model

# Charger l'image
image_path = "../assets/balle_haute.png"  # Chemin de l'image
image = cv2.imread(image_path)
assert image is not None, "Erreur : Impossible de charger l'image."

# Envoyer l'image pour analyse
print("Analyse de l'image en cours...")
results = model.predict(image_path, confidence=40, overlap=30).json()

# Vérifier si des prédictions existent
if "predictions" in results:
    for prediction in results["predictions"]:
        # Récupérer les coordonnées du rectangle de détection
        x, y, width, height = (
            prediction["x"],
            prediction["y"],
            prediction["width"],
            prediction["height"],
        )
        conf = prediction["confidence"]
        label = prediction["class"]

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

# Sauvegarder l'image annotée
output_path = "photo_annotated.png"
cv2.imwrite(output_path, image)
print(f"Image annotée sauvegardée sous : {output_path}")

# Afficher l'image annotée
cv2.imshow("Résultat", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
