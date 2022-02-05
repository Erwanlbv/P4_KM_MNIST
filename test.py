import numpy as np
import matplotlib.pyplot as plt
from variables import *
from pre_proc_programs import *
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

import time


#Pour observer les 10 premiers éléments du dataset

for i in range(10):
    print(array_train_dataset[i])
    plt.imshow(array_train_dataset[i], cmap='gray')
    plt.show()


#Pour observer les 5 premiers élements du dataset, avec un réduction puis avec réduction et lissage

"""for i in range(5):
    im = array_train_dataset[i]

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(im, cmap='gray')
    axs[1].imshow(crop_image(im, 3, 3), cmap='gray')
    axs[2].imshow(lissage(crop_image(im, 3, 3), 200), cmap='gray')

    fig.show()
"""


#Pour observer les 5 premiers éléments du dataset dont la classe est définie par nb

"""
i = 0
nb = 3
count = 0
labels = []
while count < 5:
    label = array_train_labels[i]
    if label == nb:
        labels.append(i)
        count += 1
    i += 1

fig, axs = plt.subplots(len(labels), 3)

for id, label in enumerate(labels):
    axs[id, 0].imshow(array_train_dataset[label], cmap='gray')
    axs[id, 1].imshow(lect_lig_et_col(array_train_dataset[label]), cmap='gray')
    print(lect_lig_et_col(array_train_dataset[label]))
    axs[id, 2].imshow(lissage(lect_lig_et_col(array_train_dataset[label]), 50), cmap='gray')

fig.show()
"""

#Calcul du temps d'entrainement pour Kmeans

"""
taille = 10000
train_data = np.copy(array_train_dataset[:taille])
n_clusters = 10

for i in range(2, n_clusters):
    print('Début du calcul pour le cluster n°' + str(i))
    prev_t = time.time()
    kmedoids = KMedoids(n_clusters).fit(train_data.reshape(taille, -1))
    next_t = time.time()

    compute_duration = round(next_t - prev_t, 3)
    print('-----', compute_duration)
    with open('time_res.txt', 'a') as fich:
        fich.write('\nKmedoids - Training on ' + str(i) + 'clusters without pre-processing, ' +
                   'temps de calcul : ' + str(compute_duration) + ' secondes')

"""

#Kmeans avec pré-traiement

"""
for i in range(2, n_clusters):
    train_data = np.copy(array_train_dataset[:taille])
    print('Début du calcul pour le cluster n°' + str(i))

    prev_t = time.time()
    train_data = lign_col(train_data, True, True, True)

    for j in range(len(train_data)):
        train_data[i] = (lect_lignes(train_data[j]) * True +
                              lect_colonnes(train_data[j]) * True) / (True + True)

    kmedoids = KMedoids(n_clusters).fit(train_data.reshape(taille, - 1))

    next_t = time.time()

    compute_duration = round(next_t - prev_t, 3)
    print('-----', compute_duration)
    with open('time_res.txt', 'a') as fich:
        fich.write('\nKmedoids - Training on ' + str(i) + ' clusters with pre-processing, ' +
                   'temps de calcul : ' + str(compute_duration) + ' secondes')
"""

# Pour calculer le nombre d'erreurs de classification de l'algorithme Kmeans (ou Kmedoids)

"""
taille = 5000

dataset = np.copy(array_train_dataset[:taille]).reshape(taille, -1)
labels = np.copy(array_train_labels[:taille])

for i in range(2, 100):
    print('Calcul de kmeans pour i = ' + str(i))
    kmeans = KMeans(i).fit(dataset)
    #get_label_real_class(true_labels=labels, pred_labels=kmeans.labels_)
    erreur = compute_error(true_labels=labels, pred_labels=kmeans.labels_)
    print("----- Nombre d'erreurs par classe : ", erreur)
    print("----- Nombre total d'erreurs : ", round(np.sum(erreur)/2*taille * 100), 3) #Chaque erreur étant comptée 2 fois, on divise par 2 pour obtenir le vrai taux
    print('\n')
"""

