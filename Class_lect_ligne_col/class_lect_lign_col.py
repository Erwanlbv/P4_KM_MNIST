import numpy as np
from sklearn.cluster import KMeans
import pickle
from joblib import dump, load #Optimisé pour sklearn apparemment
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


from pre_proc_programs import *
from variables import *


print(training_data.classes)

taille = 20000

train_dataset = np.copy(array_train_dataset[:taille])
train_labels = array_train_labels[:taille]

test_dataset = np.copy(array_test_dataset[:taille])
test_labels = array_test_labels[:taille]


def kmeans_lign_col():
    pre_pro_dataset = np.empty(train_dataset.shape)

    # Pré-traitement
    for i in range(len(train_dataset)):
        pre_pro_dataset[i] = lect_lig_et_col(train_dataset[i])


    # Changement de format pour la caompatibilité avec sklearn (qui veut [donnée1, donnée2, ....])
    pre_pro_dataset = pre_pro_dataset.reshape(len(pre_pro_dataset), -1)

    # On applique KMeans :
    kmeans = KMeans(n_clusters=10, n_init=10).fit(pre_pro_dataset)



    print("Centres des clusters : ", kmeans.cluster_centers_) # À afficher au format 28, 28
    print("Labels : ", kmeans.labels_) #Numpy array
    print("Nombre d'erreurs : ", np.sum(kmeans.labels_ == train_labels))

    if np.sum(kmeans.labels_ == train_labels) < taille/10:
        dump(kmeans, 'km_lign_col.joblib')


def kmean_col():
    pre_pro_dataset = np.empty(train_dataset.shape)

    for i in range(len(train_dataset)):
        pre_pro_dataset[i] = lect_colonnes(pre_pro_dataset[i])

    pre_pro_dataset = pre_pro_dataset.reshape(len(pre_pro_dataset), -1)

    #On applique KMeans :
    kmeans = KMeans(n_clusters=10, n_init=10).fit(pre_pro_dataset)


    print("Centres des clusters : ", kmeans.cluster_centers_) # À afficher au format 28, 28
    print("Labels : ", kmeans.labels_) #Numpy array
    print("Nombre d'erreurs : ", np.sum(kmeans.labels_ == train_labels))


def kmean_lignes():
    pre_pro_dataset = np.empty(train_dataset.shape)

    for i in range(len(train_dataset)):
        pre_pro_dataset[i] = lect_lignes(pre_pro_dataset[i])

    pre_pro_dataset = pre_pro_dataset.reshape(len(pre_pro_dataset), -1)

    # On applique KMeans :
    kmeans = KMeans(n_clusters=10, n_init=10).fit(pre_pro_dataset)

    print("Centres des clusters : ", kmeans.cluster_centers_)  # À afficher au format 28, 28
    print("Labels : ", kmeans.labels_)  # Numpy array
    print("Nombre d'erreurs : ", np.sum(kmeans.labels_ == train_labels))


#kmeans_lign_col()
