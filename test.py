# Importer les bibliothèques nécessaires
import numpy as np
import cv2

# Charger l'image
image = cv2.imread("tennis.jpg")

# Définir les limites de couleur pour le filtrage
boundaries = [([180, 180, 100], [255, 255, 255])]

# Parcourir les limites définies
for (lower, upper) in boundaries:
    # Créer des tableaux NumPy pour les limites
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # Trouver les couleurs dans les limites spécifiées et appliquer le masque
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

# Convertir l'image en niveaux de gris
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

# Détection des coins avec Harris
corners = cv2.cornerHarris(gray, blockSize=9, ksize=3, k=0.01)

# Normalisation des valeurs pour les rendre affichables
corners = cv2.normalize(corners, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

# Application de la méthode d'Otsu pour le seuillage
_, thresh = cv2.threshold(corners, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Dilatation de l'image seuillée
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=1)

# Trouver les contours externes
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dessiner les cercles aux centres des contours détectés
for contour in contours:
    # Calculer le moment pour obtenir le centre
    M = cv2.moments(contour)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        # Dessiner le cercle au centre du contour
        cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), -1)

# Afficher l'image résultante
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
