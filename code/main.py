import cv2
from ultralytics import YOLO
import time

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")  # Utiliser le modèle léger pré-entraîné

# Charger une vidéo
cap = cv2.VideoCapture(r"../assets/tennis.mp4")
assert cap.isOpened(), "Erreur lors de la lecture du fichier "

# Paramètres d'optimisation
# Réduire la résolution pour un traitement rapide
resize_dimensions = (640, 480)
# Traiter une frame sur 4 pour économiser les ressources
skip_frames = 2

frame_count = 0
start_time = time.time()

# Identifier l'ID de la balle de tennis dans COCO
# L'ID de la balle de tennis dans COCO
# est généralement 32 ou 41 selon les versions
sports_ball_id = 32  # ID pour "sports ball"

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Fin de la vidéo ou frame vide.")
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Redimensionner la frame
    im0 = cv2.resize(im0, resize_dimensions)

    # Détection avec YOLO
    results = model(im0)  # Détecte les objets dans l'image

    # Filtrer les résultats pour la balle de tennis (ID 32)
    for result in results:
        # Récupérer les résultats de la
        # détection des objets (boîtes de détection)
        boxes = result.boxes  # Accéder aux boîtes de détection

        # Vérifier chaque détection
        for box in boxes:
            # box est un objet Box avec les coordonnées,
            # la confiance et la classe
            # Coordonnées de la boîte de détection
            x1, y1, x2, y2 = box.xywh[0]
            conf = box.conf[0]  # Confiance de la détection
            cls = box.cls[0]  # Classe de l'objet détecté

            if (
                int(cls) == sports_ball_id
            ):  # Vérifier si l'objet est une balle de tennis
                print(
                    f"Balle de tennis detectee a : x1={x1}, y1={y1}, x2={x2}, y2={y2}"
                )
                # Dessiner la boîte de détection sur l'image
                cv2.rectangle(
                    im0, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )

    # Afficher l'image avec la détection
    cv2.imshow("Balle de Tennis Détectée", im0)

    # Appuyer sur "q" pour quitter
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Fin du traitement
end_time = time.time()
processing_time = end_time - start_time
print(f"Temps de traitement : {processing_time:.2f} secondes")

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
