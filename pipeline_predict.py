# pipeline_predict.py : chaîne complète sur un dossier d'images.
# Étape 1 : le détecteur trouve les avions. Étape 2 : le classifieur détermine le type de chacun.
# Résultat : un fichier JSON (et, en option, des images annotées).

import argparse, glob, os                        # argparse : options ; glob : liste de fichiers ; os : chemins
import cv2                                       # lecture et écriture d'images
import common as c                               # nos fonctions partagées

def main():
    ap = argparse.ArgumentParser(description="Détection puis classification -> JSON")   # prépare la lecture des options
    ap.add_argument("--images", default=c.IMAGES)              # dossier des images à analyser
    ap.add_argument("--detection", default="models/detection.pt")   # fichier du détecteur
    ap.add_argument("--classification", default="models/classification.pt")   # fichier du classifieur
    ap.add_argument("--out", default="resultats/annotations_predites.json")   # fichier JSON de sortie
    ap.add_argument("--score", type=float, default=0.5, help="score minimal d'une détection")   # seuil de confiance du détecteur
    ap.add_argument("--images-annotees", default="", help="dossier où écrire les images dessinées (optionnel)")   # si renseigné, on enregistre aussi les images avec les boîtes
    a = ap.parse_args()                                        # lit les options

    det, clf = c.charger_detecteur(a.detection), c.charger_classifieur(a.classification)   # charge les deux modèles
    noms = sorted(os.path.basename(p) for e in ("*.jpg", "*.jpeg", "*.png")
                  for p in glob.glob(os.path.join(a.images, e)))   # liste triée des images du dossier (.jpg, .jpeg, .png)
    if a.images_annotees:                                      # si on veut enregistrer les images dessinées...
        os.makedirs(a.images_annotees, exist_ok=True)          # ...on crée le dossier

    res = {}                                                   # résultats : {image: [(classe, x1, y1, x2, y2), ...]}
    for n in noms:                                             # pour chaque image du dossier
        img = cv2.imread(os.path.join(a.images, n))            # lit l'image (format BGR)
        boxes, _ = c.detecter(img, det, a.score)               # étape 1 : détecte les avions (le score n'est pas utilisé ici, d'où le _)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)             # version RGB de l'image, attendue par le classifieur
        probs = c.classer([c.square_crop(rgb, *b) for b in boxes], clf)   # étape 2 : découpe chaque avion détecté et calcule ses probabilités de classe
        res[n] = [(c.CLASSES[int(p.argmax())], *b) for b, p in zip(boxes, probs)]   # pour chaque avion : classe la plus probable + boîte détectée
        if a.images_annotees:                                  # si on veut les images dessinées
            out = img.copy()                                   # copie de l'image pour dessiner dessus
            for b, p in zip(boxes, probs):                     # pour chaque avion détecté
                c.dessiner(out, b, f"{c.CLASSES[int(p.argmax())]} {p.max()*100:.0f}%", p.max() >= 0.6)   # dessine la boîte avec la classe et la confiance ; vert si confiance >= 60 %, sinon orange
            cv2.imwrite(os.path.join(a.images_annotees, n), out)   # enregistre l'image annotée

    c.ecrire_json(a.out, res)                                  # écrit le fichier JSON de résultats
    print(len(noms), "images,", sum(len(v) for v in res.values()), "avions ->", a.out)   # nombre d'images traitées et d'avions trouvés

if __name__ == "__main__":                                     # exécuté seulement quand on lance ce fichier directement
    main()                                                     # lance la chaîne complète