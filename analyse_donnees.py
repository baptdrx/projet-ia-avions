# analyse_donnees.py : analyse du corpus (nombre d'images et d'avions, classes, tailles des boîtes...).
# C'est la première étape du projet : comprendre les données avant d'entraîner quoi que ce soit.

import os                                        # chemins et fichiers
import cv2, pandas as pd                         # cv2 : images ; pandas : tableaux de données
import matplotlib.pyplot as plt                  # tracé de graphiques
from PIL import Image                            # lecture de la taille des images
import common as c                               # nos fonctions partagées

ann = c.lire_annotations(c.XML, c.IMAGES)                      # lit toutes les annotations : une ligne par avion
ann["w"], ann["h"] = ann["x2"] - ann["x1"], ann["y2"] - ann["y1"]   # ajoute la largeur (w) et la hauteur (h) de chaque boîte, en pixels
noms = ann["image"].unique()                                   # liste des noms d'images (sans doublons)
tailles = pd.Series([Image.open(os.path.join(c.IMAGES, n)).size for n in noms])   # taille (largeur, hauteur) de chaque image

print("Images :", len(noms), "| Avions annotés :", len(ann))   # nombre d'images et d'avions annotés
print("\nObjets par classe :\n", ann["classe"].value_counts())   # nombre d'avions de chaque classe (pour voir le déséquilibre)
print("\nTailles d'images (largeur, hauteur) :\n", tailles.value_counts().head(10))   # tailles d'images les plus fréquentes
print("\nTaille des boîtes (px) :\n", ann[["w", "h"]].describe())   # statistiques sur la taille des avions : moyenne, minimum, maximum...
print("\nObjets par image :\n", ann.groupby("image").size().describe())   # statistiques sur le nombre d'avions par image

fig, ax = plt.subplots(1, 3, figsize=(16, 4))                  # prépare une figure avec 3 graphiques côte à côte
ann["classe"].value_counts().plot.bar(ax=ax[0], title="Objets par classe")   # graphique 1 : barres, nombre d'avions par classe
ann["w"].plot.hist(bins=40, ax=ax[1], title="Largeur des boîtes (px)")   # graphique 2 : histogramme des largeurs de boîtes
ann.groupby("image").size().plot.hist(bins=20, ax=ax[2], title="Objets par image")   # graphique 3 : histogramme du nombre d'avions par image
plt.tight_layout(); plt.show()                                 # ajuste les marges et affiche les graphiques

def show(nom):                                                  # contrôle visuel : boîtes sur une image
    img = cv2.cvtColor(cv2.imread(os.path.join(c.IMAGES, nom)), cv2.COLOR_BGR2RGB)   # lit l'image et la convertit en RGB pour l'affichage
    for r in ann[ann["image"] == nom].itertuples():            # pour chaque avion annoté de cette image
        cv2.rectangle(img, (r.x1, r.y1), (r.x2, r.y2), (255, 0, 0), 3)   # dessine sa boîte en rouge (épaisseur 3)
        cv2.putText(img, str(r.classe), (r.x1, max(r.y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)   # écrit sa classe au-dessus de la boîte
    plt.figure(figsize=(10, 10)); plt.imshow(img); plt.axis("off"); plt.show()   # affiche l'image sans les axes

show(ann["image"].iloc[0])                                      # affiche la première image du jeu de données avec ses boîtes