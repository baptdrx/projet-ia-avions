# classification_train.py : entraîne le classifieur (ResNet50) qui détermine le type d'un avion à partir d'une petite image.

import argparse, os, time                        # argparse : options en ligne de commande ; os : chemins ; time : chronomètre
import numpy as np, torch, torch.nn as nn        # numpy : tableaux ; torch : réseaux de neurones ; nn : couches et fonctions de perte
from torch.utils.data import Dataset, DataLoader # Dataset : décrit nos données ; DataLoader : les fournit par lots
from torchvision import transforms               # transformations d'images (data augmentation)
import common as c                               # nos fonctions partagées

class Crops(Dataset):
    # Jeu de données : une petite image (crop) par avion, avec son numéro de classe.
    def __init__(self, X, y):
        self.X, self.y = X, y                                  # X : les crops ; y : leurs classes
        self.aug = transforms.Compose([                        # data augmentation : modifications aléatoires appliquées à chaque passage
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),   # retournement horizontal et vertical au hasard (un avion peut avoir n'importe quelle orientation)
            transforms.RandomRotation(180), transforms.ColorJitter(brightness=0.2, contrast=0.2)])   # rotation aléatoire jusqu'à 180° ; variation de luminosité et de contraste de 20 %
        self.norm = transforms.Normalize(c.MEAN.tolist(), c.STD.tolist())   # normalisation des couleurs comme pour ImageNet
    def __len__(self):
        return len(self.X)                                     # nombre d'exemples
    def __getitem__(self, i):
        x = self.aug(torch.from_numpy(self.X[i]).permute(2, 0, 1))   # prend le crop numéro i, le convertit en tenseur (canaux, hauteur, largeur) et applique l'augmentation
        return self.norm(x.float() / 255), int(self.y[i])      # valeurs ramenées entre 0 et 1, normalisées ; renvoie l'image et sa classe

def main():
    ap = argparse.ArgumentParser(description="Entraîne le classifieur de types d'avions (ResNet50).")   # prépare la lecture des options
    ap.add_argument("--images", default=c.IMAGES)              # dossier des images
    ap.add_argument("--xml", default=c.XML)                    # dossier des annotations
    ap.add_argument("--out", default="models/classification.pt")   # fichier où sauvegarder le modèle entraîné
    ap.add_argument("--epochs-head", type=int, default=5)      # nombre d'epochs de la phase 1 (dernière couche seule)
    ap.add_argument("--epochs-fine", type=int, default=10)     # nombre d'epochs de la phase 2 (réseau entier)
    ap.add_argument("--batch", type=int, default=32)           # nombre d'images traitées en même temps
    ap.add_argument("--seed", type=int, default=42)            # graine du hasard (pour retrouver les mêmes résultats)
    a = ap.parse_args()                                        # lit les options
    torch.manual_seed(a.seed); np.random.seed(a.seed)          # fixe le hasard de torch et de numpy

    ann = c.lire_annotations(a.xml, a.images)                  # lit toutes les annotations
    ann = ann[~ann["image"].isin(c.images_eval(ann))]             # retire les images d'évaluation : on n'entraîne que sur la partie "train"
    X, y, _ = c.construire_crops(ann, a.images)                # découpe un crop par avion d'entraînement
    print("Device :", c.DEVICE, "|", len(X), "crops d'entraînement")   # affiche le matériel utilisé et le nombre de crops
    loader = DataLoader(Crops(X, y), batch_size=a.batch, shuffle=True, num_workers=0)   # fournit les données par lots, dans un ordre aléatoire à chaque epoch

    counts = np.bincount(y, minlength=len(c.CLASSES))             # nombre d'exemples de chaque classe (contre le déséquilibre des classes)
    w = torch.tensor(counts.sum() / (len(c.CLASSES) * np.maximum(counts, 1)), dtype=torch.float32).to(c.DEVICE)   # poids de chaque classe : plus la classe est rare, plus son poids est grand
    crit = nn.CrossEntropyLoss(weight=w)                       # fonction de perte (mesure l'erreur de classification), pondérée par classe
    model = c.construire_classifieur(pretrained=True).to(c.DEVICE)   # crée le ResNet50 avec les poids d'ImageNet et l'envoie sur le GPU
    cuda = c.DEVICE.type == "cuda"                             # True si on utilise le GPU
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)        # outil du calcul en précision mixte (plus rapide, moins de mémoire)

    def fit(epochs, lr):
        # Entraîne le modèle pendant "epochs" passages complets, avec le learning rate "lr" (taille des pas d'apprentissage).
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)   # optimiseur Adam : ne met à jour que les paramètres autorisés à apprendre
        for e in range(epochs):                                # une epoch = un passage complet sur toutes les données
            model.train(); t0, tot, ok = time.time(), 0, 0     # mode entraînement ; chrono ; cumul de la perte ; nombre de bonnes prédictions
            for x, t in loader:                                # pour chaque lot : x = images, t = vraies classes
                x, t = x.to(c.DEVICE), t.to(c.DEVICE)          # envoie le lot sur le GPU
                with torch.autocast("cuda", enabled=cuda):     # calcul en précision mixte
                    out = model(x); loss = crit(out, t)        # prédiction du modèle, puis erreur par rapport à la vérité
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()   # remet les gradients à zéro ; calcule les gradients (rétropropagation) ; met à jour les poids ; ajuste l'échelle de la précision mixte
                tot += loss.item() * len(x); ok += (out.argmax(1) == t).sum().item()   # cumule la perte et compte les bonnes réponses (classe prédite = classe réelle)
            print(f"ep {e+1}/{epochs} | loss {tot/len(y):.3f} | acc {ok/len(y):.3f} | {time.time()-t0:.0f}s")   # affiche la perte moyenne, la précision et la durée de l'epoch

    for p in model.parameters(): p.requires_grad = False          # phase 1 : on gèle tout le réseau (il n'apprend plus)...
    for p in model.fc.parameters(): p.requires_grad = True        # ...sauf la dernière couche (fc), la seule qui apprend
    print("--- Phase 1 : tête seule ---"); fit(a.epochs_head, 1e-3)   # entraîne la dernière couche avec un learning rate de 0,001
    for p in model.parameters(): p.requires_grad = True           # phase 2 : on dégèle tout le réseau
    print("--- Phase 2 : fine-tuning ---"); fit(a.epochs_fine, 1e-4)   # entraîne tout le réseau avec un learning rate 10 fois plus petit, pour ne pas abîmer les poids pré-entraînés

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)  # crée le dossier de sortie s'il n'existe pas
    torch.save({"state_dict": model.state_dict(), "classes": c.CLASSES, "arch": c.ARCH}, a.out)   # sauvegarde les poids, la liste des classes et l'architecture
    print("Modèle enregistré :", a.out)                        # confirme la sauvegarde

if __name__ == "__main__":                                     # exécuté seulement quand on lance ce fichier directement
    main()                                                     # lance l'entraînement