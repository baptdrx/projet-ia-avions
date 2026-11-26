# classification_eval.py : évalue le classifieur sur les avions annotés
# (la boîte de chaque avion est connue, on prédit seulement son type).

import argparse                                  # lecture des options en ligne de commande
import matplotlib.pyplot as plt                  # tracé de graphiques (matrice de confusion)
from sklearn.metrics import classification_report, confusion_matrix   # métriques : précision, rappel, F1, matrice de confusion
import common as c                               # nos fonctions partagées

def main():
    ap = argparse.ArgumentParser(description="Évalue le classifieur : écrit un JSON et calcule précision, rappel, F1.")   # prépare la lecture des options
    ap.add_argument("--model", default="models/classification.pt")   # modèle à évaluer
    ap.add_argument("--images", default=c.IMAGES)              # dossier des images
    ap.add_argument("--xml", default=c.XML, help="vérité terrain (boîtes + classes) : dossier de XML ou fichier JSON")   # annotations de référence
    ap.add_argument("--subset", choices=["eval", "all"], default="eval")   # eval : seulement la partie évaluation ; all : toutes les images
    ap.add_argument("--out", default="resultats/predictions_classification.json")   # fichier JSON de sortie
    ap.add_argument("--matrice", default="resultats/confusion_classification.png")   # image de la matrice de confusion
    a = ap.parse_args()                                        # lit les options

    ann = c.lire_annotations(a.xml, a.images)                  # lit les annotations (vérité terrain)
    if a.subset == "eval":                                     # si on évalue seulement la partie évaluation...
        ann = ann[ann["image"].isin(c.images_eval(ann))]       # ...on ne garde que ses images
    X, y, ann = c.construire_crops(ann, a.images)                 # un crop par avion annoté
    probs = c.classer(X, c.charger_classifieur(a.model))       # le classifieur donne, pour chaque crop, une probabilité par classe

    res = {}                                                   # résultats : {image: [(classe, x1, y1, x2, y2), ...]}
    for r, p in zip(ann.itertuples(), probs):                     # pour chaque avion : r = ses annotations, p = ses probabilités
        res.setdefault(r.image, []).append((c.CLASSES[int(p.argmax())], r.x1, r.y1, r.x2, r.y2))   # ajoute à la liste de l'image : classe la plus probable + boîte annotée
    c.ecrire_json(a.out, res)                                  # écrit le fichier JSON
    print(len(ann), "avions classés ->", a.out, "\n")          # affiche le nombre d'avions classés

    y_pred = probs.argmax(1)                                   # numéro de la classe prédite pour chaque avion (probabilité maximale)
    labels = list(range(len(c.CLASSES)))                       # liste des numéros de classes : 0...19
    print(classification_report(y, y_pred, labels=labels, target_names=c.CLASSES, digits=3, zero_division=0))   # tableau par classe : précision, rappel, F1, nombre d'exemples

    cm = confusion_matrix(y, y_pred, labels=labels)            # matrice de confusion : ligne = vraie classe, colonne = classe prédite
    plt.figure(figsize=(9, 8)); plt.imshow(cm, cmap="Blues"); plt.colorbar()   # dessine la matrice en dégradé de bleus, avec une légende de couleurs
    plt.xticks(labels, c.CLASSES, rotation=90); plt.yticks(labels, c.CLASSES)   # noms des classes sur les axes
    plt.xlabel("Prédit"); plt.ylabel("Réel"); plt.title("Matrice de confusion")   # titres des axes et du graphique
    plt.tight_layout(); plt.savefig(a.matrice, dpi=120)        # ajuste les marges et enregistre l'image
    print("Matrice de confusion :", a.matrice)                 # affiche le chemin de l'image

if __name__ == "__main__":                                     # exécuté seulement quand on lance ce fichier directement
    main()                                                     # lance l'évaluation