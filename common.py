# common.py : fonctions partagées par tous les scripts du projet
# (lecture des annotations, découpe des avions, chargement des modèles, prédiction).

import os, json                                  # os : chemins et fichiers ; json : lecture / écriture de fichiers JSON
import xml.etree.ElementTree as ET               # lecteur de fichiers XML (les annotations sont au format Pascal VOC)
import cv2, numpy as np, pandas as pd, torch, torch.nn as nn   # cv2 : images ; numpy : tableaux ; pandas : tableaux de données ; torch : réseaux de neurones
from sklearn.model_selection import train_test_split           # coupe les données en deux groupes (entraînement / évaluation)
from torchvision import models                   # réseaux classiques déjà écrits (ResNet)
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights   # détecteur Faster R-CNN et ses poids pré-entraînés
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor   # dernière couche du détecteur (prédit la classe et la boîte)

# ---------- Réglages généraux ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # carte graphique (cuda) si elle existe, sinon processeur
CLASSES = sorted(f"A{i}" for i in range(1, 21))               # les 20 classes A1...A20 triées par ordre alphabétique (A1, A10, A11...) ; cet ordre doit rester celui de l'entraînement
ARCH = "resnet50"                                              # architecture du classifieur (ResNet à 50 couches)
IMAGES = os.environ.get("IMAGES_DIR", "student-dataset 2/images")      # dossier des images (modifiable par variable d'environnement, utile avec Docker)
XML = os.environ.get("XML_DIR", "student-dataset 2/annotations")       # dossier des annotations XML
SIZE, MARGIN = 224, 0.15                                       # taille des petites images données au classifieur (224x224) ; marge de 15 % autour de chaque avion
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)       # moyenne des couleurs d'ImageNet (le réseau pré-entraîné s'y attend)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)        # écart-type des couleurs d'ImageNet

# ---------- Données ----------
def trouver_image(img_dir, stem):
    # Cherche l'image qui correspond à une annotation : même nom, extension jpg / jpeg / png.
    for ext in (".jpg", ".jpeg", ".png", ".JPG"):              # essaie chaque extension possible
        if os.path.exists(os.path.join(img_dir, stem + ext)):  # le fichier existe-t-il ?
            return stem + ext                                  # oui : on renvoie son nom complet (ex. "12.jpg")
    return None                                                # aucune image trouvée

def lire_annotations_json(chemin):
    # Lit un fichier JSON d'annotations {"1.jpg": [{"class": "A2", "coordinates": {...}}]} et le transforme en tableau pandas.
    data = json.load(open(chemin, encoding="utf-8"))           # charge tout le fichier JSON dans un dictionnaire Python
    rows = [{"image": nom, "classe": o["class"],               # une ligne par avion : nom de l'image, classe de l'avion...
             "x1": int(o["coordinates"]["xmin"]), "y1": int(o["coordinates"]["ymin"]),   # ...coin haut-gauche de la boîte (entiers)
             "x2": int(o["coordinates"]["xmax"]), "y2": int(o["coordinates"]["ymax"])}   # ...coin bas-droit de la boîte
            for nom, objs in data.items() for o in objs]       # on parcourt toutes les images, puis tous les avions de chaque image
    return pd.DataFrame(rows)                                  # liste de lignes -> tableau pandas

def lire_annotations(source, img_dir):
    # Lit les annotations : soit un fichier .json, soit un dossier de fichiers .xml.
    # Renvoie un tableau avec une ligne par avion : image, classe, x1, y1, x2, y2.
    if str(source).lower().endswith(".json"):                  # la source est-elle un fichier JSON ?
        ann = lire_annotations_json(source)                    # oui : on le lit avec la fonction ci-dessus
        return ann[ann["image"].map(lambda n: os.path.exists(os.path.join(img_dir, n)))].reset_index(drop=True)   # on ne garde que les avions dont l'image existe dans le dossier d'images
    rows = []                                                  # liste qui va contenir une ligne par avion
    for f in sorted(os.listdir(source)):                       # parcourt les fichiers du dossier d'annotations, dans l'ordre alphabétique
        if not f.endswith(".xml"):                             # ignore ce qui n'est pas un fichier XML
            continue
        nom = trouver_image(img_dir, os.path.splitext(f)[0])   # retrouve l'image associée (même nom, sans l'extension .xml)
        if nom is None:                                        # pas d'image correspondante
            print("Image introuvable pour", f); continue       # on le signale et on passe au fichier suivant
        root = ET.parse(os.path.join(source, f)).getroot()     # ouvre le fichier XML ; root = racine de l'arbre XML
        for obj in root.findall("object"):                     # chaque balise <object> est un avion annoté
            bb = obj.find("bndbox")                            # balise qui contient les coordonnées de la boîte
            rows.append({"image": nom, "classe": obj.findtext("name"),   # nom de l'image et classe de l'avion (balise <name>)
                         "x1": int(float(bb.findtext("xmin"))), "y1": int(float(bb.findtext("ymin"))),   # coin haut-gauche (converti en nombre entier)
                         "x2": int(float(bb.findtext("xmax"))), "y2": int(float(bb.findtext("ymax")))})  # coin bas-droit
    return pd.DataFrame(rows)                                  # liste de lignes -> tableau pandas

def images_eval(ann, frac=0.3, seed=42):
    """Noms des images de la partie évaluation (split par image, stratifié sur la classe la plus rare)."""
    freq = ann["classe"].value_counts()                        # nombre d'avions de chaque classe dans tout le jeu de données
    lab = (ann.assign(freq=ann["classe"].map(freq)).sort_values("freq")   # ajoute à chaque avion la fréquence de sa classe, puis trie du plus rare au plus fréquent
              .groupby("image")["classe"].first())             # pour chaque image, garde la classe la plus rare qu'elle contient : c'est son "étiquette" pour le découpage
    _, ev = train_test_split(lab.index.to_numpy(), test_size=frac,   # coupe la liste des images : 70 % entraînement (ignoré, noté _) / 30 % évaluation (ev)
                             stratify=lab.to_numpy(), random_state=seed)   # stratify : mêmes proportions de classes dans les deux groupes ; seed : tirage reproductible
    return set(ev)                                             # renvoie l'ensemble des noms d'images d'évaluation

def square_crop(img, x1, y1, x2, y2, margin=MARGIN):
    """Crop carré avec marge, complété en noir au bord, redimensionné en 224x224."""
    H, W = img.shape[:2]                                       # hauteur et largeur de l'image
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2                      # centre de la boîte de l'avion
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)            # côté du carré : le plus grand côté de la boîte, agrandi de 15 % de chaque côté
    a, b = int(round(cx - side / 2)), int(round(cy - side / 2))   # coin haut-gauche du carré
    c, d = int(round(cx + side / 2)), int(round(cy + side / 2))   # coin bas-droit du carré
    pl, pt = max(0, -a), max(0, -b)                            # pixels à ajouter à gauche et en haut si le carré dépasse de l'image
    pr, pb = max(0, c - W), max(0, d - H)                      # idem à droite et en bas
    crop = img[max(0, b):min(H, d), max(0, a):min(W, c)]       # découpe la zone, limitée à l'intérieur de l'image
    if pl or pt or pr or pb:                                   # si le carré dépassait de l'image...
        crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)   # ...on complète avec des bords noirs
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)   # redimensionne en 224x224 pixels (INTER_AREA : bonne méthode pour réduire)

def construire_crops(ann, img_dir):
    """Un crop RGB par avion annoté. Retourne (crops, indices de classe, annotations triées)."""
    ann = ann.sort_values("image", kind="stable").reset_index(drop=True)   # trie les avions par nom d'image (tri stable) pour avoir un ordre fixe
    X = []                                                     # liste des petites images (crops)
    for nom, g in ann.groupby("image", sort=True):             # pour chaque image, g = tableau de ses avions
        img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, nom)), cv2.COLOR_BGR2RGB)   # lit l'image (OpenCV la lit en BGR) et la convertit en RGB
        X += [square_crop(img, r.x1, r.y1, r.x2, r.y2) for r in g.itertuples()]         # découpe un crop pour chaque avion de l'image
    y = np.array([CLASSES.index(k) for k in ann["classe"]])    # transforme chaque nom de classe en numéro (A1 -> 0, A10 -> 1, ...)
    return np.stack(X), y, ann                                 # crops empilés en un seul tableau, numéros de classes, annotations triées

def iou(a, b):
    # IoU = surface commune / surface totale de deux boîtes : 1 = boîtes identiques, 0 = boîtes qui ne se touchent pas.
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])   # coordonnées du rectangle d'intersection
    inter = max(0, x2 - x1) * max(0, y2 - y1)                  # surface de l'intersection (0 si les boîtes ne se recouvrent pas)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter   # surface de la réunion = surface A + surface B - intersection
    return inter / union if union > 0 else 0.0                 # rapport intersection / réunion (0 si la division est impossible)

# ---------- Sortie JSON et dessin ----------
def ecrire_json(chemin, resultats):
    # Écrit {image: [(classe, x1, y1, x2, y2), ...]} dans un fichier JSON de la forme {"class": ..., "coordinates": {...}}.
    sortie = {nom: [{"class": k, "coordinates": {"xmin": int(round(x1)), "ymin": int(round(y1)),   # pour chaque image : liste d'avions avec classe et coordonnées arrondies en entiers
                                                 "xmax": int(round(x2)), "ymax": int(round(y2))}}
                    for (k, x1, y1, x2, y2) in objs]           # un dictionnaire par avion
              for nom, objs in resultats.items()}              # une entrée par image
    os.makedirs(os.path.dirname(chemin) or ".", exist_ok=True) # crée le dossier de sortie s'il n'existe pas
    json.dump(sortie, open(chemin, "w", encoding="utf-8"), indent=2)   # écrit le fichier JSON (indent=2 : mise en forme lisible)

def dessiner(img, box, label, fiable=True):
    # Dessine une boîte et son étiquette sur l'image.
    x1, y1, x2, y2 = [int(v) for v in box]                     # coordonnées de la boîte en nombres entiers
    col = (0, 200, 0) if fiable else (0, 140, 255)             # couleur (BGR) : vert si la classe est fiable, orange sinon
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)             # trace le rectangle (épaisseur 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)   # mesure la taille du texte (largeur tw, hauteur th)
    y0 = max(y1, th + 6)                                       # position verticale du texte, sans sortir de l'image par le haut
    cv2.rectangle(img, (x1, y0 - th - 6), (x1 + tw + 4, y0), col, -1)   # rectangle plein de la même couleur, qui sert de fond au texte
    cv2.putText(img, label, (x1 + 2, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)   # écrit l'étiquette en noir

# ---------- Modèles ----------
def construire_detecteur(pretrained=False):
    m = fasterrcnn_resnet50_fpn(                               # crée un Faster R-CNN (ResNet50 + FPN)
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,   # poids pré-entraînés sur COCO si pretrained=True (Transfer Learning)
        weights_backbone=None, min_size=800, max_size=800, box_detections_per_img=300)   # images traitées en 800x800 ; jusqu'à 300 détections par image
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)   # remplace la dernière couche par une couche à 2 sorties
    return m                                                   # 2 classes : fond + avion

def charger_detecteur(chemin):
    m = construire_detecteur()                                 # crée le détecteur (sans poids pré-entraînés)
    m.load_state_dict(torch.load(chemin, map_location=DEVICE)) # charge nos poids entraînés depuis le fichier
    return m.to(DEVICE).eval()                                 # envoie le modèle sur GPU / CPU et le passe en mode évaluation (pas d'entraînement)

def construire_classifieur(arch=ARCH, pretrained=False):
    if arch == "resnet50":                                     # ResNet à 50 couches
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)   # avec les poids d'ImageNet si pretrained=True
    else:                                                      # sinon ResNet à 18 couches (ancienne version)
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))           # remplace la dernière couche (fc) par une couche à 20 sorties, une par classe
    return m                                                   # renvoie le réseau

def charger_classifieur(chemin):
    ck = torch.load(chemin, map_location=DEVICE)               # charge le fichier du modèle
    nouveau = "state_dict" in ck                                 # nouveau format : poids + architecture
    m = construire_classifieur(ck.get("arch", "resnet18") if nouveau else "resnet18")   # recrée le bon réseau (architecture lue dans le fichier, ResNet18 pour l'ancien format)
    m.load_state_dict(ck["state_dict"] if nouveau else ck)     # charge les poids appris
    return m.to(DEVICE).eval()                                 # envoie sur GPU / CPU et passe en mode évaluation

# ---------- Inférence (utilisation des modèles) ----------
def detecter(img_bgr, modele, seuil=0.5):
    """Image BGR -> (boîtes xyxy, scores) des avions détectés."""
    x = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(DEVICE)   # image BGR -> RGB -> tenseur (canaux, hauteur, largeur) -> valeurs entre 0 et 1 -> envoyée sur le GPU
    with torch.no_grad():                                      # pas de calcul de gradient (on n'entraîne pas) : plus rapide
        o = modele([x])[0]                                     # le détecteur prédit les boîtes ; o = résultat de la première (et seule) image
    k = o["scores"] >= seuil                                   # garde seulement les détections dont le score dépasse le seuil
    return o["boxes"][k].cpu().numpy(), o["scores"][k].cpu().numpy()   # boîtes et scores retenus, ramenés sur le CPU en tableaux numpy

def classer(crops_rgb, modele, batch=64):
    """Crops RGB -> matrice de probabilités (N x 20)."""
    out = []                                                   # liste des résultats de chaque lot
    for i in range(0, len(crops_rgb), batch):                  # traite les crops par lots de 64
        x = np.stack([(c.astype(np.float32) / 255 - MEAN) / STD for c in crops_rgb[i:i + batch]])   # pour chaque crop : valeurs entre 0 et 1, puis normalisation (moyenne et écart-type d'ImageNet) ; empile le lot
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(DEVICE) # tableau numpy -> tenseur torch ; réordonne les axes (lot, canaux, hauteur, largeur) ; envoie sur le GPU
        with torch.no_grad():                                  # pas de gradient (inférence)
            out.append(torch.softmax(modele(x), dim=1).cpu().numpy())   # le réseau donne un score par classe ; softmax les transforme en probabilités (somme = 1)
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))   # recolle tous les lots ; tableau vide s'il n'y a aucun crop