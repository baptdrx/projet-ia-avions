# detection_train.py : entraîne le détecteur Faster R-CNN, qui trouve la position des avions dans une image.

import argparse, os, time                        # argparse : options en ligne de commande ; os : chemins ; time : chronomètre
import cv2, numpy as np, torch                   # cv2 : lecture des images ; numpy : tableaux ; torch : réseaux de neurones
from torch.utils.data import Dataset, DataLoader # Dataset : décrit nos données ; DataLoader : les fournit par lots
import common as c                               # nos fonctions partagées

class Avions(Dataset):
    # Jeu de données : une image complète et la liste des boîtes de ses avions.
    def __init__(self, ann, img_dir):
        self.noms = sorted(ann["image"].unique())              # liste triée des noms d'images
        self.boxes = {n: g[["x1", "y1", "x2", "y2"]].values.astype("float32")   # pour chaque image : tableau des boîtes (x1, y1, x2, y2)
                      for n, g in ann.groupby("image")}        # (une entrée par image du tableau d'annotations)
        self.dir = img_dir                                     # dossier des images
    def __len__(self):
        return len(self.noms)                                  # nombre d'images du jeu de données
    def __getitem__(self, i):
        n = self.noms[i]                                       # nom de l'image numéro i
        img = torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(self.dir, n)), cv2.COLOR_BGR2RGB))   # lit l'image, la convertit en RGB puis en tenseur
        img = img.permute(2, 0, 1).float().div(255)            # réordonne les axes (canaux, hauteur, largeur) et ramène les valeurs entre 0 et 1
        H, W = img.shape[1:]                                   # hauteur et largeur de l'image
        b = torch.from_numpy(self.boxes[n].copy())             # boîtes de cette image (copie, pour pouvoir les modifier)
        if torch.rand(1) < 0.5:                                # une fois sur deux : retournement horizontal (data augmentation)
            img = img.flip(-1); b[:, [0, 2]] = W - b[:, [2, 0]]   # retourne l'image ; nouveau x1 = W - ancien x2, nouveau x2 = W - ancien x1
        if torch.rand(1) < 0.5:                                # une fois sur deux : retournement vertical
            img = img.flip(-2); b[:, [1, 3]] = H - b[:, [3, 1]]   # retourne l'image ; nouveau y1 = H - ancien y2, nouveau y2 = H - ancien y1
        b[:, [0, 2]] = b[:, [0, 2]].clamp(0, W); b[:, [1, 3]] = b[:, [1, 3]].clamp(0, H)   # force les boîtes à rester dans l'image
        b = b[((b[:, 2] - b[:, 0]) > 2) & ((b[:, 3] - b[:, 1]) > 2)]   # supprime les boîtes trop petites (2 pixels ou moins)
        return img, {"boxes": b, "labels": torch.ones(len(b), dtype=torch.int64)}   # image + cibles : les boîtes et l'étiquette 1 (= avion ; 0 = fond)

def collate(batch):
    return tuple(zip(*batch))                                  # regroupe un lot en (images...), (cibles...) ; nécessaire car chaque image a un nombre différent de boîtes

def main():
    ap = argparse.ArgumentParser(description="Entraîne le détecteur d'avions (Faster R-CNN).")   # prépare la lecture des options
    ap.add_argument("--images", default=c.IMAGES)              # dossier des images
    ap.add_argument("--xml", default=c.XML)                    # dossier des annotations
    ap.add_argument("--out", default="models/detection.pt")    # fichier où sauvegarder le modèle entraîné
    ap.add_argument("--epochs", type=int, default=15)          # nombre d'epochs (passages complets sur les données)
    ap.add_argument("--batch", type=int, default=4)            # nombre d'images traitées en même temps
    ap.add_argument("--lr", type=float, default=0.005)         # learning rate : taille des pas d'apprentissage
    ap.add_argument("--seed", type=int, default=42)            # graine du hasard (pour retrouver les mêmes résultats)
    a = ap.parse_args()                                        # lit les options
    torch.manual_seed(a.seed); np.random.seed(a.seed)          # fixe le hasard de torch et de numpy

    ann = c.lire_annotations(a.xml, a.images)                  # lit toutes les annotations
    ann = ann[~ann["image"].isin(c.images_eval(ann))]            # retire les images d'évaluation : on n'entraîne que sur la partie "train"
    loader = DataLoader(Avions(ann, a.images), batch_size=a.batch, shuffle=True,
                        num_workers=2, collate_fn=collate)     # fournit les images par lots, dans un ordre aléatoire ; 2 processus préparent les données en parallèle
    print("Device :", c.DEVICE, "|", len(loader.dataset), "images d'entraînement")   # affiche le matériel utilisé et le nombre d'images

    model = c.construire_detecteur(pretrained=True).to(c.DEVICE)  # crée le détecteur avec les poids COCO (Transfer Learning) et l'envoie sur le GPU
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=a.lr, momentum=0.9, weight_decay=5e-4)   # optimiseur SGD : momentum = inertie ; weight_decay = limite la taille des poids
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)   # planning : le learning rate diminue en cloche (cosinus) pendant l'entraînement
    cuda = c.DEVICE.type == "cuda"                             # True si on utilise le GPU
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)        # outil du calcul en précision mixte (plus rapide, moins de mémoire)
    it = 0                                                     # compteur d'itérations (nombre de lots déjà traités)
    for ep in range(a.epochs):                                 # boucle sur les epochs
        model.train(); t0, tot = time.time(), 0                # mode entraînement ; chrono ; cumul de la perte
        for imgs, tgts in loader:                              # pour chaque lot : imgs = images, tgts = boîtes et étiquettes correctes
            if it < 200:                                       # pendant les 200 premiers lots : montée progressive du learning rate (warmup)
                for g in opt.param_groups: g["lr"] = a.lr * (it + 1) / 200   # le learning rate augmente petit à petit jusqu'à sa valeur normale
            imgs = [i.to(c.DEVICE) for i in imgs]              # envoie les images sur le GPU
            tgts = [{k: v.to(c.DEVICE) for k, v in t.items()} for t in tgts]   # envoie aussi les cibles sur le GPU
            with torch.autocast("cuda", enabled=cuda):         # calcul en précision mixte
                loss = sum(model(imgs, tgts).values())         # en entraînement, le modèle renvoie plusieurs erreurs (position, classe...) ; on les additionne
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()   # remet les gradients à zéro ; calcule les gradients (rétropropagation) ; met à jour les poids ; ajuste l'échelle de la précision mixte
            tot += loss.item(); it += 1                        # cumule la perte et compte un lot de plus
        sched.step()                                           # fait avancer le planning du learning rate (une fois par epoch)
        print(f"ep {ep+1}/{a.epochs} | loss {tot/len(loader):.3f} | {time.time()-t0:.0f}s")   # affiche le numéro d'epoch, la perte moyenne et la durée

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)  # crée le dossier de sortie s'il n'existe pas
    torch.save(model.state_dict(), a.out)                      # sauvegarde les poids appris (dernier modèle, sans sélection du meilleur epoch)
    print("Modèle enregistré :", a.out)                        # confirme la sauvegarde

if __name__ == "__main__":                                     # exécuté seulement quand on lance ce fichier directement
    main()                                                     # lance l'entraînement