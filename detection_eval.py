# detection_eval.py : évalue le détecteur (précision, rappel, F1) et écrit ses détections dans un fichier JSON.

import argparse, glob, os                        # argparse : options ; glob : liste de fichiers ; os : chemins
import cv2                                       # lecture des images
import common as c                               # nos fonctions partagées

def main():
    ap = argparse.ArgumentParser(description="Évalue le détecteur : écrit un JSON et calcule précision, rappel, F1.")   # prépare la lecture des options
    ap.add_argument("--model", default="models/detection.pt")  # modèle à évaluer
    ap.add_argument("--images", default=c.IMAGES)              # dossier des images
    ap.add_argument("--xml", default=c.XML,
                    help="vérité terrain : dossier de XML ou fichier JSON ; '' pour juste produire le JSON")   # annotations de référence (vide : pas de calcul de score)
    ap.add_argument("--subset", choices=["eval", "all"], default="eval",
                    help="eval = partie évaluation du dataset ; all = toutes les images du dossier")   # quelles images évaluer
    ap.add_argument("--out", default="resultats/predictions_detection.json")   # fichier JSON de sortie
    ap.add_argument("--score", type=float, default=0.5, help="score minimal d'une détection")   # seuil de confiance du détecteur
    ap.add_argument("--iou", type=float, default=0.5, help="IoU minimal pour qu'une détection soit juste")   # recouvrement minimal avec la vraie boîte
    a = ap.parse_args()                                        # lit les options

    if a.xml:                                                  # si on a une vérité terrain (annotations)...
        ann = c.lire_annotations(a.xml, a.images)              # ...on la lit
        if a.subset == "eval":                                 # option : seulement la partie évaluation
            ann = ann[ann["image"].isin(c.images_eval(ann))]   # on ne garde que les images d'évaluation
        noms = sorted(ann["image"].unique())                   # liste des images à traiter
        gts = {n: g[["x1", "y1", "x2", "y2"]].values for n, g in ann.groupby("image")}   # vraies boîtes, rangées par image
    else:                                                      # pas d'annotations : on traite toutes les images du dossier
        noms = sorted(os.path.basename(p) for e in ("*.jpg", "*.jpeg", "*.png")
                      for p in glob.glob(os.path.join(a.images, e)))   # liste des fichiers .jpg / .jpeg / .png

    modele = c.charger_detecteur(a.model)                      # charge le détecteur entraîné
    preds = {}                                                 # prédictions : {image: [{"bbox": ..., "score": ...}]}
    for n in noms:                                             # pour chaque image
        boxes, scores = c.detecter(cv2.imread(os.path.join(a.images, n)), modele, a.score)   # détecte les avions (boîtes + scores)
        preds[n] = [{"bbox": [float(v) for v in b], "score": float(s)} for b, s in zip(boxes, scores)]   # mémorise chaque boîte avec son score
    c.ecrire_json(a.out, {n: [("avion", *p["bbox"]) for p in preds[n]] for n in noms})   # écrit le JSON (le détecteur ne connaît pas le type : classe "avion")
    print(len(noms), "images,", sum(len(v) for v in preds.values()), "avions détectés ->", a.out)   # nombre d'images et d'avions détectés

    if a.xml:                                                  # comparaison avec la vérité terrain
        TP = FP = FN = 0                                       # vrais positifs, faux positifs, faux négatifs
        for n in noms:                                         # pour chaque image
            gt, used, tp = gts[n], set(), 0                    # gt : vraies boîtes ; used : vraies boîtes déjà associées ; tp : détections justes de l'image
            for p in preds[n]:                                 # déjà triées par score décroissant
                best, j = 0.0, -1                              # meilleur recouvrement trouvé, et vraie boîte correspondante
                for k, g in enumerate(gt):                     # on compare à chaque vraie boîte...
                    if k not in used:                          # ...pas encore associée à une autre détection
                        v = c.iou(p["bbox"], g)                # recouvrement (IoU) entre la détection et la vraie boîte
                        if v > best: best, j = v, k            # on garde la meilleure
                if best >= a.iou:                              # recouvrement suffisant : la détection est juste
                    used.add(j); tp += 1                       # marque la vraie boîte comme utilisée et compte une détection juste
            TP += tp; FP += len(preds[n]) - tp; FN += len(gt) - tp   # TP : justes ; FP : détections sans avion correspondant ; FN : avions non détectés
        P, R = TP / max(TP + FP, 1), TP / max(TP + FN, 1)      # précision = TP / (TP + FP) ; rappel = TP / (TP + FN)
        F1 = 2 * P * R / max(P + R, 1e-9)                      # F1 = moyenne harmonique de la précision et du rappel
        print(f"TP {TP} | FP {FP} | FN {FN}  (score >= {a.score}, IoU >= {a.iou})")   # affiche les comptes
        print(f"Précision {P:.3f} | Rappel {R:.3f} | F1 {F1:.3f}")   # affiche les scores

if __name__ == "__main__":                                     # exécuté seulement quand on lance ce fichier directement
    main()                                                     # lance l'évaluation