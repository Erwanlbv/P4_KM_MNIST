import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

from pre_proc_programs import *
from variables import *


def kmeans_lign_col(dataset, n_cluster, L, norm_bool, liss_bool, crop_bool):

    # Pré-traitement
    pre_pro_dataset = pre_pro(dataset, norm_bool, liss_bool, crop_bool)

    for i in range(len(pre_pro_dataset)): # L[0] correspond au pré-traitement avec les lignes, L[1] correspond à celui des colonnes
        pre_pro_dataset[i] = (lect_lignes(pre_pro_dataset[i]) * L[0] +
                              lect_colonnes(pre_pro_dataset[i]) * L[1]) / (L[0] + L[1])

    # Changement de format pour compatibilité avec sklearn
    pre_pro_dataset = pre_pro_dataset.reshape(len(pre_pro_dataset), -1)

    # Entrainement
    kmeans = KMeans(n_clusters=n_cluster, n_init=10).fit(pre_pro_dataset)

    return pre_pro_dataset, kmeans


def kmedoids_lign_col(dataset, n_cluster, L, norm_bool, liss_bool, crop_bool):

    # Pré-traitement
    pre_pro_dataset = pre_pro(dataset, norm_bool, liss_bool, crop_bool)

    for i in range(len(pre_pro_dataset)): # L[0] correspond au pré-traitement avec les lignes, L[1] correspond à celui des colonnes
        pre_pro_dataset[i] = (lect_lignes(pre_pro_dataset[i]) * L[0] +
                              lect_colonnes(pre_pro_dataset[i]) * L[1]) / (L[0] + L[1])

    # Changement de format (compatibilité avec KMEDOIS
    pre_pro_dataset = pre_pro_dataset.reshape(len(pre_pro_dataset), -1)

    # Entrainement
    kmedoids = KMedoids(n_clusters=n_cluster, metric='euclidean', method='alternate', init='random').fit(pre_pro_dataset)

    return pre_pro_dataset, kmedoids


def visu_simple():
    dataset = np.copy(array_train_dataset)
    fig, axs = plt.subplots(5, 2)

    for i in range(5):
        axs[i, 0].imshow(dataset[i], cmap='gray')
        axs[i, 1].imshow(lect_lig_et_col(dataset[i]), cmap='gray')

    fig.show()

#visu_simple()
