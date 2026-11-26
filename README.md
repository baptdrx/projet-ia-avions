# Détection et classification d'avions sur images satellites

Projet du module « Introduction à l'IA ».

L'application localise les avions dans une image satellite, puis détermine leur type (20 classes, `A1` à `A20`).

Auteurs : **Baptiste Darroux** et **Thomas Duprat** et **Matthias Delarue**

## Architecture

Deux modèles enchaînés :

```
image 800x800
   |
   v
Faster R-CNN (ResNet50-FPN)  ->  boîtes des avions (une seule classe : "avion")
   |
   v
découpe carrée de chaque boîte (marge de 15 %, redimensionnée en 224x224)
   |
   v
ResNet50  ->  type d'avion (A1 à A20) + confiance
   |
   v
JSON au format demandé / image annotée / interface Streamlit
```

| Brique | Choix | Pourquoi |
|---|---|---|
| Détection | Faster R-CNN ResNet50-FPN, poids COCO (Transfer Learning), 2 classes (fond, avion) | détecteur *two-stage* précis, le FPN aide pour les petits objets (~100 px) |
| Classification | ResNet50, poids ImageNet, nouvelle couche à 20 sorties | Transfer Learning : le jeu d'entraînement est petit |
| Données | split 70 % / 30 % **par image**, stratifié sur la classe la plus rare de l'image (graine 42) | éviter qu'un même avion ou une même image soit en entraînement et en évaluation |

Entraînement :

- **Détection** : 15 epochs, batch 4, SGD (lr 0,005, momentum 0,9), planification cosinus, montée progressive du learning rate, retournements horizontal et vertical, précision mixte.
- **Classification** : deux phases (tête seule : 5 epochs à 1e-3, puis fine-tuning de tout le réseau : 10 epochs à 1e-4), Adam, batch 32, perte pondérée par classe (déséquilibre 107 à 785 exemplaires), augmentation (retournements, rotations quelconques, luminosité et contraste).
- Dans les deux cas, on garde le **dernier** modèle (pas de sélection sur le jeu d'évaluation).

## Structure du dépôt

```
common.py                  fonctions partagées (lecture des annotations, split, découpe, modèles, inférence)
detection_train.py         entraîne le détecteur          (images + XML -> models/detection.pt)
detection_eval.py          évalue le détecteur            (modèle + images -> JSON, précision / rappel / F1)
classification_train.py    entraîne le classifieur        (images + XML -> models/classification.pt)
classification_eval.py     évalue le classifieur          (modèle + images -> JSON, précision / rappel / F1)
pipeline_predict.py        chaîne complète                (images -> JSON au format du prof)
analyse_donnees.py         analyse du corpus
app.py                     interface Streamlit
Dockerfile, requirements.txt, .dockerignore
models/                    detection.pt, classification.pt  (Git LFS)
resultats/                 JSON de prédictions, matrice de confusion, capture de la page de validation
```

## Installation

Python 3.12 recommandé.

```bash
pip install -r requirements.txt
```

Pour utiliser un GPU NVIDIA, installer PyTorch avec CUDA (voir pytorch.org), par exemple :

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Données

Les données ne sont pas versionnées. Structure attendue :

```
student-dataset 2/
  images/         images .jpg
  annotations/    annotations Pascal VOC (.xml)
```

Les chemins peuvent être changés avec les variables d'environnement `IMAGES_DIR` et `XML_DIR`, ou avec les options `--images` et `--xml` des scripts.

## Utilisation

```bash
# Entraînement
python detection_train.py
python classification_train.py

# Évaluation (écrit un JSON au format du prof et affiche précision, rappel, F1)
python detection_eval.py
python classification_eval.py

# Chaîne complète sur un dossier d'images -> JSON au format du prof
python pipeline_predict.py --images "eval-dataset/images" --out resultats/annotations_test.json

# Interface graphique
python -m streamlit run app.py
```

L'option `--xml` des scripts d'évaluation accepte un dossier de XML ou un fichier JSON au format du prof, avec `--subset all` pour utiliser toutes les images.

## Format des JSON

```json
{
  "1.jpg": [
    {"class": "A2", "coordinates": {"xmin": 485, "ymin": 427, "xmax": 554, "ymax": 500}}
  ]
}
```

- `detection_eval.py` : la classe est `"avion"` (le détecteur ne connaît pas le type).
- `classification_eval.py` : boîtes de la vérité terrain, classe prédite.
- `pipeline_predict.py` : boîtes détectées, classes prédites (c'est le fichier à envoyer à la page de validation).

## Docker

```bash
docker build -t avions-app .
docker run --rm -p 8501:8501 -v "<dossier du dataset>:/data" avions-app
```

L'application est ensuite disponible sur 'http://localhost:8501'. L'image contient le code et les deux modèles, PyTorch en version processeur ; le dataset est monté au lancement (`/data/images`, `/data/annotations`).

## Résultats

### Évaluation interne (400 images, 2331 avions, jamais vues à l'entraînement)

| | Précision | Rappel | F1 |
|---|---|---|---|
| Détection (IoU >= 0,5, score >= 0,5) | 0,988 | 0,993 | 0,991 |
| Classification (boîtes annotées) | accuracy 0,958 | | F1 macro 0,933 |
| Chaîne complète | | | classification 0,948, global 0,939 |

### Jeu de test (2511 images, 14471 avions)

| | Valeur |
|---|---|
| Overall F1 | **0,7726** |
| Détection : précision / rappel / F1 | 0,9878 / 0,9730 / **0,9804** |
| Classification : précision / rappel / F1 | 0,7940 / 0,7821 / **0,7880** |

## Analyse et limites

- La **détection généralise très bien** (F1 de 0,98 sur des images nouvelles).
- La **classification chute** de 0,95 (nos images) à 0,79 . Notre évaluation interne était trop optimiste : nos images d'évaluation sont probablement proches de celles d'entraînement (mêmes scènes), alors que le jeu de test en contient de nouvelles. Par exemple, A13 a un rappel de 0,99 en interne et de 0,46 sur le jeu du prof.
- Les types **A15 et A5** sont confondus de façon systématique (rappel de A15 d'environ 0,4 à 0,5), probablement parce que les avions se ressemblent.
- Pistes non réalisées : réduire la marge de découpe pour limiter l'influence du fond, augmentation de données plus forte, entraînement sur toutes les images, test-time augmentation.
