# app.py : interface web (Streamlit) pour tester la chaîne complète : détection des avions, puis classification.
# Deux fonctions : "Image aléatoire" (tire une image au hasard et l'analyse)
# et "Traiter tout le dossier" (analyse toutes les images, écrit le JSON, les images annotées et un CSV).
# Lancement : python -m streamlit run app.py
# Important : Streamlit relance tout le script à chaque action de l'utilisateur (clic, curseur déplacé...).

import os, glob, random                          # os : chemins ; glob : liste de fichiers ; random : tirage au sort
import cv2, pandas as pd                         # cv2 : images ; pandas : tableaux de données
import streamlit as st                           # bibliothèque qui construit la page web
import common as c                               # nos fonctions partagées

MODELE_DET = os.environ.get("MODELE_DET", "models/detection.pt")        # fichier du détecteur (modifiable par variable d'environnement)
MODELE_CLS = os.environ.get("MODELE_CLS", "models/classification.pt")   # fichier du classifieur
SORTIE = os.environ.get("SORTIE_DIR", "sortie_annotee")                 # dossier où sont écrits les résultats du traitement complet
DOSSIER_DEFAUT = os.environ.get("DEMO_DIR", c.IMAGES)                   # dossier d'images proposé au départ (par défaut : celui du dataset)

st.set_page_config(page_title="Détection d'avions", layout="wide")      # titre de l'onglet du navigateur et page en pleine largeur
st.title("Détection et classification d'avions sur images satellites")  # grand titre de la page
st.caption("Faster R-CNN localise les avions, puis un réseau ResNet détermine leur type (A1 à A20).")   # petit texte sous le titre

@st.cache_resource                               # garde le résultat en mémoire : les modèles ne sont chargés qu'une seule fois
def charger_modeles():
    return c.charger_detecteur(MODELE_DET), c.charger_classifieur(MODELE_CLS)   # charge le détecteur et le classifieur

@st.cache_data                                   # garde le résultat en mémoire (données simples)
def noms_evaluation():
    return sorted(c.images_eval(c.lire_annotations(c.XML, c.IMAGES)))   # liste triée des noms d'images de la partie évaluation

def analyser(img_bgr, seuil_det, seuil_conf):
    """Image BGR -> (image annotée, lignes du tableau, objets au format JSON des annotations)."""
    det, clf = charger_modeles()                               # récupère les deux modèles (déjà en mémoire après le premier appel)
    boxes, scores = c.detecter(img_bgr, det, seuil_det)        # étape 1 : détecte les avions (boîtes et scores)
    out, rows, objets = img_bgr.copy(), [], []                 # image sur laquelle on dessine ; lignes du tableau ; avions pour le JSON
    if len(boxes):                                             # si au moins un avion a été détecté
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)         # version RGB de l'image, attendue par le classifieur
        probs = c.classer([c.square_crop(rgb, *b) for b in boxes], clf)   # étape 2 : découpe chaque avion et calcule ses probabilités de classe
        for k, (b, s, p) in enumerate(zip(boxes, scores, probs), 1):   # pour chaque avion (k = numéro à partir de 1, b = boîte, s = score, p = probabilités)
            i = int(p.argmax()); conf = float(p[i]); ok = conf >= seuil_conf   # i = numéro de la classe la plus probable ; conf = sa probabilité ; ok = type fiable ?
            c.dessiner(out, b, f"#{k} {c.CLASSES[i]} {conf*100:.0f}%" + ("" if ok else " ?"), ok)   # dessine la boîte avec le numéro, la classe et la confiance (un "?" si incertain)
            rows.append({"N°": k, "Type": c.CLASSES[i], "Confiance du type": round(conf, 3),
                         "Confiance de détection": round(float(s), 3), "Fiable": "oui" if ok else "non"})   # ajoute une ligne au tableau affiché
            objets.append((c.CLASSES[i], *b))                  # ajoute l'avion (classe + boîte) pour l'export JSON
    return out, rows, objets                                   # image annotée, lignes du tableau, avions pour le JSON

def afficher(img, seuil_det, seuil_conf):
    # Analyse une image et affiche le résultat dans la page.
    with st.spinner("Analyse en cours..."):                    # affiche un message d'attente pendant le calcul
        out, rows, _ = analyser(img, seuil_det, seuil_conf)    # lance l'analyse (le 3e résultat n'est pas utilisé ici)
    col_img, col_res = st.columns([3, 2])                      # deux colonnes : image (3/5 de la largeur) et résultats (2/5)
    with col_img:                                              # colonne de gauche
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Vert : type fiable | Orange : incertain (?)")   # affiche l'image annotée (convertie en RGB pour l'affichage)
    with col_res:                                              # colonne de droite
        if rows:                                               # si au moins un avion a été détecté
            df = pd.DataFrame(rows)                            # tableau des avions
            c1, c2 = st.columns(2)                             # deux petites colonnes pour les compteurs
            c1.metric("Avions détectés", len(df))              # nombre d'avions détectés
            c2.metric("Types incertains", int((df["Fiable"] == "non").sum()))   # nombre d'avions dont le type est incertain
            st.dataframe(df, hide_index=True)                  # affiche le tableau détaillé
        else:                                                  # aucun avion détecté
            st.warning("Aucun avion détecté. Essayez de baisser le seuil de détection.")   # message d'avertissement

# ---------- Barre latérale (réglages) ----------
st.sidebar.header("Réglages")                                  # titre de la barre latérale
seuil_det = st.sidebar.slider("Seuil de détection", 0.10, 0.95, 0.50, 0.05,
                              help="Plus il est haut, moins il y a de fausses détections.")   # curseur : score minimal pour garder une détection
seuil_conf = st.sidebar.slider("Seuil de confiance du type", 0.30, 0.95, 0.60, 0.05,
                               help="En dessous, le type d'avion est marqué incertain (?).")   # curseur : confiance minimale pour que le type soit "fiable"
st.sidebar.write("Matériel utilisé :", "GPU (CUDA)" if c.DEVICE.type == "cuda" else "processeur")   # indique si le calcul se fait sur le GPU ou le processeur

# ---------- Choix du dossier d'images ----------
dossier = st.text_input("Dossier d'images", DOSSIER_DEFAUT)    # champ de texte : dossier où lire les images
noms = sorted(os.path.basename(p) for e in ("*.jpg", "*.jpeg", "*.png")
              for p in glob.glob(os.path.join(dossier, e)))    # liste triée des images du dossier
if dossier == c.IMAGES and os.path.isdir(c.XML):               # si on est sur le dossier du dataset et que les annotations existent...
    ev = set(noms_evaluation())                                # ...ensemble des images d'évaluation (non vues à l'entraînement)
    noms = [n for n in noms if n in ev]                        # on ne garde que celles-là
if not noms:                                                   # aucune image trouvée
    st.error("Aucune image (.jpg, .jpeg, .png) trouvée dans ce dossier."); st.stop()   # affiche une erreur et arrête le script
st.write(f"{len(noms)} images disponibles.")                   # affiche le nombre d'images disponibles

# ---------- Image aléatoire ----------
def tirer_au_hasard():
    # Choisit une image au hasard, différente de l'image actuelle.
    autres = [n for n in noms if n != st.session_state.get("choix")] or noms   # toutes les images sauf l'actuelle (st.session_state = mémoire de la session)
    st.session_state["choix"] = random.choice(autres)          # mémorise l'image tirée au sort

if st.session_state.get("choix") not in noms:                  # premier affichage, ou changement de dossier
    tirer_au_hasard()                                          # on tire une première image au hasard
st.button("Image aléatoire", on_click=tirer_au_hasard)         # bouton : au clic, tire une nouvelle image
choix = st.session_state["choix"]                              # image actuellement choisie
st.caption(f"Image : {choix}")                                 # affiche son nom
img = cv2.imread(os.path.join(dossier, choix))                 # lit l'image choisie
if img is None:                                                # fichier illisible
    st.error("Image illisible."); st.stop()                    # affiche une erreur et arrête le script
afficher(img, seuil_det, seuil_conf)                           # analyse et affiche l'image

# ---------- Traitement de tout le dossier ----------
st.divider()                                                   # ligne de séparation
if st.button("Traiter tout le dossier"):                       # bouton : analyse toutes les images du dossier
    os.makedirs(SORTIE, exist_ok=True)                         # crée le dossier de sortie
    lignes, resultats, barre = [], {}, st.progress(0.0)        # lignes du tableau CSV ; avions pour le JSON ; barre de progression
    for n, nom in enumerate(noms, 1):                          # pour chaque image (n = compteur à partir de 1)
        im = cv2.imread(os.path.join(dossier, nom))            # lit l'image
        if im is not None:                                     # si elle est lisible
            out, rows, objets = analyser(im, seuil_det, seuil_conf)   # analyse l'image
            cv2.imwrite(os.path.join(SORTIE, nom), out)        # enregistre l'image annotée
            lignes += [{"Image": nom, **r} for r in rows]      # ajoute les lignes de cette image au tableau global
            resultats[nom] = objets                            # mémorise les avions de cette image pour le JSON
        barre.progress(n / len(noms))                          # fait avancer la barre de progression
    chemin_json = os.path.join(SORTIE, "annotations_predites.json")   # chemin du fichier JSON de résultats
    c.ecrire_json(chemin_json, resultats)                      # écrit le JSON au format des annotations
    df = pd.DataFrame(lignes)                                  # tableau global de tous les avions
    df.to_csv(os.path.join(SORTIE, "predictions.csv"), index=False)   # enregistre ce tableau en CSV
    st.success(f"{len(noms)} images traitées, {len(df)} avions détectés. "
               f"Images annotées, annotations_predites.json et predictions.csv dans : {SORTIE}")   # message de réussite
    if len(df):                                                # s'il y a au moins un avion
        st.bar_chart(df["Type"].value_counts())                # graphique : nombre d'avions par type
    st.download_button("Télécharger annotations_predites.json (format JSON des annotations)",
                       open(chemin_json, "rb").read(), "annotations_predites.json", "application/json")   # bouton pour télécharger le JSON