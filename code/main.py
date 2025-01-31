import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from ultralytics import YOLO

# Dimensions du terrain (ajuste selon tes besoins)
terrain_largeur = 105  # en mètres
terrain_hauteur = 68   # en mètres

# Points de référence pour la transformation
points_src = np.array([[423, 185], [231, 501], [831, 161], [1047, 503]], dtype='float32')
points_dst = np.array([[0, 0], [0, terrain_hauteur], [terrain_largeur, 0], [terrain_largeur, terrain_hauteur]], dtype='float32')

# Calculer la matrice de transformation
matrix = cv2.getPerspectiveTransform(points_src, points_dst)

# Initialisation du modèle YOLO
model_path = "../code/best.pt"
model = YOLO(model_path)

# Charger la vidéo
video_path = "../Vidéos/MatchV1.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

# Création de la figure et des axes Matplotlib
fig, ax = plt.subplots()
ax.set_xlim(0, terrain_largeur)
ax.set_ylim(0, terrain_hauteur)

# Dessiner le terrain
terrain = patches.Rectangle((0, 0), terrain_largeur, terrain_hauteur, linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(terrain)

# Point pour représenter la balle
balle, = ax.plot([], [], 'yo', markersize=20)  # 'yo' : jaune, cercle, taille augmentée

# Texte pour afficher les coordonnées de la balle
coord_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

# Lire les propriétés de la vidéo
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Vidéo chargée : {total_frames} frames à {fps:.2f} FPS.")

frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fin de la vidéo ou erreur lors de la lecture.")
        break

    results = model.predict(frame, conf=0.4, iou=0.3, verbose=False)

    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1 = (
                box.xyxy[0][0].item(),  # Coordonnée x1
                box.xyxy[0][1].item(),  # Coordonnée y1
            )

            # Transformer les coordonnées de la balle
            points = np.array([[x1, y1]], dtype='float32')
            transformed_points = cv2.perspectiveTransform(np.array([points]), matrix)[0][0]
            x_transformed, y_transformed = transformed_points

            # Mettre à jour la position de la balle
            balle.set_data([x_transformed], [y_transformed])
            coord_text.set_text(f'Coordonnées de la balle: ({x_transformed:.2f}, {y_transformed:.2f})')
            plt.pause(0.1)  # Pause pour actualiser l'affichage

    frame_counter += 1

cap.release()
plt.show()