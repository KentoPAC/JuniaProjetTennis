# Détection du Terrain de Tennis

Ce module permet de détecter les points clés d'un terrain de tennis à partir d'une vidéo. Il utilise une approche basée sur un réseau de neurones convolutifs (CNN) pour détecter 14 points caractéristiques du terrain.

## Architecture

### 1. Modèle Neural (tracknet.py)

Le modèle `BallTrackerNet` est un réseau de neurones basé sur une architecture encodeur-décodeur :
- Encodeur : série de convolutions et max pooling pour extraire les caractéristiques
- Décodeur : upsampling et convolutions pour reconstruire les heatmaps
- 18 couches de convolution au total
- Sortie : 15 heatmaps (14 points du terrain + 1 pour la balle)

### 2. Post-traitement (postprocess.py)

Deux fonctions principales :
- `postprocess()` : Convertit les heatmaps en coordonnées (x,y) en utilisant la détection de cercles de Hough
- `refine_kps()` : Affine la position des points en utilisant la détection de lignes

### 3. Homographie (homography.py)

Permet de corriger la perspective du terrain en :
- Utilisant des configurations prédéfinies de 4 points
- Calculant la meilleure matrice d'homographie
- Transformant tous les points pour avoir une vue "normalisée"

### 4. Court Reference (court_reference.py)

Définit le modèle de référence du terrain avec :
- Les coordonnées de tous les points clés
- Les différentes configurations de 4 points possibles
- Les dimensions standards du terrain

## Utilisation

```python
from terrain.infer_in_video import infer_terrain

points = infer_terrain(
    model_path="path/to/model.pth",
    video_path="input.mp4",
    output_json="output.json",
    duration=5.0,
    use_refine_kps=True,
    use_homography=True
)
```

## Forces

1. **Robustesse** :
   - Fonctionne sur différents angles de caméra
   - Gestion des occlusions partielles
   - Utilisation de plusieurs frames pour plus de stabilité

2. **Précision** :
   - Post-traitement sophistiqué (détection de cercles + lignes)
   - Correction de perspective via homographie
   - Raffinement des points par intersection de lignes

3. **Facilité d'utilisation** :
   - Interface simple avec un seul point d'entrée
   - Sortie JSON standardisée
   - Vidéo annotée générée automatiquement

## Considérations de Performance

1. **Ressources requises** :
   - Nécessite un GPU pour des performances optimales
   - Post-traitement potentiellement intensif
   - Utilisation importante de mémoire

2. **Optimisation des ressources** :
   - La détection du terrain n'est effectuée qu'une seule fois au début de l'analyse
   - Les coordonnées sont mémorisées pour toute la durée de la vidéo
   - Cette approche permet d'économiser significativement les ressources sur le long terme

## Limitations

1. **Limites de détection** :
   - Sensible aux conditions d'éclairage
   - Difficulté avec les angles extrêmes
   - La qualité de la détection initiale est cruciale car utilisée pour toute la vidéo

2. **Dépendances** :
   - Nécessite plusieurs bibliothèques (PyTorch, OpenCV, numpy)
   - Modèle pré-entraîné requis
   - Sensible aux versions des dépendances


## Recommandations d'utilisation

1. **Vidéo d'entrée** :
   - Résolution minimale recommandée : 640x360
   - Durée d'analyse conseillée : 5 secondes
   - Éviter les mouvements de caméra rapides

2. **Paramètres optimaux** :
   - Activer `use_refine_kps` pour plus de précision
   - Ajuster la durée selon la qualité de détection

3. **Post-traitement** :
   - Filtrer les détections aberrantes
   - Moyenner les positions sur plusieurs frames
   - Vérifier la cohérence géométrique des points

## Extensions possibles


1. **Nouvelles fonctionnalités** :
   - Détection automatique du type de terrain
   - Support de caméras multiples
   - Calibration automatique

2. **Robustesse** :
   - Meilleure gestion des conditions difficiles
   - Auto-adaptation aux différents styles de terrain
   - Détection de la qualité des points détectés
```
