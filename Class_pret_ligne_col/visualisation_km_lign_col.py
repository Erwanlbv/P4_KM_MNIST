import time
import matplotlib.pyplot as plt

from variables import *
from class_lect_lign_col import kmeans_lign_col
from scores import compute_scores, compute_error
from general_visualizations import display_histos

TAILLE = [100, 5000, 10000]
MIN_CLUSTERS = 2
MAX_CLUSTERS = 40


def kmeans_class_col_visu():
    norm_bool, liss_bool, crop_bool = True, True, True
    iner_bool, sil_bool, davies_bool = True, True, True
    L = [True, True]
    # Pour afficher les histogrammes
    disp_histos = False

    fig, axs = plt.subplots(3, figsize=(13, 7))
    fig.suptitle("Kmeans avec pré-traitement " +
                 '\n Pré-traitement : ' + str(L) +
                 '; Normalisation : ' + str(norm_bool) +
                 ', Lissage : ' + str(liss_bool) +
                 ', Réduction : ' + str(crop_bool), fontsize=12)

    for taille in TAILLE:
        dataset = np.copy(array_train_dataset[:taille])
        res = np.zeros((MAX_CLUSTERS - MIN_CLUSTERS + 1, 3))

        for i in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
            print('Début des calculs pour ' + str(i) + ' clusters')
            _, km = kmeans_lign_col(np.copy(dataset), i, L, norm_bool=norm_bool, liss_bool=liss_bool, crop_bool=crop_bool)
            res[i - MIN_CLUSTERS] = compute_scores(_, km, inertia_bool=iner_bool, sil_bool=sil_bool, davies_bool=True)
            print('Résultats :' + str(res[i - MIN_CLUSTERS]))

            # Pour afficher les histogrammes (nombre d'images par classes à la fin):
            if disp_histos:
                display_histos(km.labels_, array_train_labels[:taille])

        for id, ax in enumerate(axs.flat):
            ax.set_title("Inertie du modèle : " * (id == 0) + 'Score Silouhette : ' * (id == 1) + 'Davies Score' * (id == 2))
            ax.plot(range(MIN_CLUSTERS, MAX_CLUSTERS + 1), res[:, id], label=(str(taille) + ' images'))
            if id < 2:
                ax.get_xaxis().set_visible(False)
                ax.set_xlabel('Nombre de clusters')
            ax.legend()

    fig.show()

kmeans_class_col_visu()