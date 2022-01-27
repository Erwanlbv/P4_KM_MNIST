import numpy as np
from sklearn.cluster import KMeans
import pickle
from joblib import dump, load #Optimisé pour sklearn apparemment
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


from pre_proc_programs import *
from variables import *

TAILLE = 1000
MIN_CLUSTERS = 2
MAX_CLUSTERS = 15


def km_only(dataset, n_cluster, norm_bool, liss_bool, crop_bool):

    # Pré-traitement classique
    pre_pro_dataset = pre_pro(dataset, norm_bool, liss_bool, crop_bool)
    pre_pro_dataset = pre_pro_dataset.reshape(len(pre_pro_dataset), -1) # Adapte le format à celui demandé par sklearn.

    # Pré-traitement supplémentaire :


    # Entrainement
    print("KM Only - Fin du pré-traitement, début de l'entrainement..")
    kmeans = KMeans(n_cluster).fit(pre_pro_dataset)
    print("KM Only - Fin de l'entrainement")

    return pre_pro_dataset, kmeans


def kmeans_lign_col(norm_bool, liss_bool):
    pre_pro_dataset = np.empty(train_dataset.shape)

    # Pré-traitement
    pre_pro_dataset = pre_pro(pre_pro_dataset, norm_bool, liss_bool)

    for i in range(len(train_dataset)):
        pre_pro_dataset[i] = lect_lig_et_col(train_dataset[i])

    # Changement de format pour la compatibilité avec sklearn (qui veut [donnée1, donnée2, ....])
    pre_pro_dataset = pre_pro_dataset.reshape(len(pre_pro_dataset), -1)

    # On applique KMeans :
    kmeans = KMeans(n_clusters=10, n_init=10).fit(pre_pro_dataset)


    print("Centres des clusters : ", kmeans.cluster_centers_) # À afficher au format 28, 28
    print("Labels : ", kmeans.labels_) #Numpy array
    print("Nombre d'erreurs : ", np.sum(kmeans.labels_ == train_labels))

    if np.sum(kmeans.labels_ == train_labels) < taille/10:
        dump(kmeans, 'km_lign_col.joblib')


def kmean_col(norm_bool, liss_bool):
    pre_pro_dataset = np.empty(train_dataset.shape)

    # Pré-traitement
    pre_pro_dataset = pre_pro(pre_pro_dataset, norm_bool, liss_bool)

    for i in range(len(train_dataset)):
        pre_pro_dataset[i] = lect_colonnes(pre_pro_dataset[i])

    pre_pro_dataset = pre_pro_dataset.reshape(len(pre_pro_dataset), -1)

    #On applique KMeans :
    print("Fin du pré-traitement, début de l'entrainement..")
    kmeans = KMeans(n_clusters=10, n_init=10).fit(pre_pro_dataset)
    print("Fin de l'entraiment, l'inertie est de " + str(kmeans.inertia_))


    print("Centres des clusters : ", kmeans.cluster_centers_) # À afficher au format 28, 28
    print("Labels : ", kmeans.labels_) #Numpy array
    print("Nombre d'erreurs : ", np.sum(kmeans.labels_ == train_labels))


def kmean_lignes(norm_bool, liss_bool):
    pre_pro_dataset = np.empty(train_dataset.shape)

    # Pré-traitement
    pre_pro_dataset = pre_pro(pre_pro_dataset, norm_bool, liss_bool)

    for i in range(len(train_dataset)):
        pre_pro_dataset[i] = lect_lignes(pre_pro_dataset[i])

    pre_pro_dataset = pre_pro_dataset.reshape(len(pre_pro_dataset), -1)

    # On applique KMeans :
    kmeans = KMeans(n_clusters=10, n_init=10).fit(pre_pro_dataset)

    print("Centres des clusters : ", kmeans.cluster_centers_)  # À afficher au format 28, 28
    print("Labels : ", kmeans.labels_)  # Numpy array
    print("Nombre d'erreurs : ", np.sum(kmeans.labels_ == train_labels))


#kmeans_lign_col()
