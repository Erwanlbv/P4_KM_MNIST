import matplotlib.pyplot as plt
import numpy as np

import joblib as jb
from class_kmeans_only import kmeans_only, train_kms, km_only
from scores import compute_scores
from variables import *

TAILLE = 10000
MIN_CLUSTERS = 2
MAX_CLUSTERS = 15


def visualisation_km_only():

    dataset = np.copy(array_train_dataset[:TAILLE])
    res = np.zeros((MAX_CLUSTERS - MIN_CLUSTERS + 1, 3))
    norm_bool, liss_bool, crop_bool = True, True, True

    for i in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
        print('Début des calculs pour ' + str(i) + ' clusters')
        _, km = km_only(np.copy(dataset), i, norm_bool=norm_bool, liss_bool=liss_bool, crop_bool=crop_bool)
        res[i - MIN_CLUSTERS] = compute_scores(_, km, norm_bool, liss_bool, crop_bool)
        print('Résultats :' + str(res[i - MIN_CLUSTERS]))

    fig, axs = plt.subplots(3, figsize=(13, 7))
    fig.suptitle("Kmeans only - Nombre d'images utilisées : " + str(TAILLE) +
                 '\n Normalisation : ' + str(norm_bool) +
                 ' Lissage : ' + str(liss_bool) +
                 ' Réduction : ' + str(crop_bool), fontsize=15)

    for id, ax in enumerate(axs.flat):
        ax.set_title("Inertie du modèle : " * (id == 0) + 'Score Silouhette : ' * (id == 1) + 'Davies Score' * (id == 2))
        ax.plot(range(MIN_CLUSTERS, MAX_CLUSTERS + 1), res[:, id])
        if id < 2:
            ax.get_xaxis().set_visible(False)

    fig.show()
    plt.pause(0.01)

#Problème : Association groupe du modèle - chiffre en réalité

visualisation_km_only()