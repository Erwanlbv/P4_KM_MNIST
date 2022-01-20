import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
from sklearn.cluster import KMeans
from joblib import dump, load


def kmeans_only(train_dataset, train_labels, taille, min, max):

    train_dataset = train_dataset[:taille]
    train_labels = train_labels[:taille]

    err = []
    inertie = []

    for i in range(min, max):
        print('Nombre de cluster : ' +str(i))
        #On adapte le format à KM
        train_dataset = train_dataset.reshape(len(train_dataset), -1)

        # Regroupement
        kmeans = KMeans(n_clusters=i).fit(train_dataset)

        #print("Centres des clusters : ", kmeans.cluster_centers_) # À afficher au format 28, 28
        #print("Labels : ", kmeans.labels_) #Numpy array
        print("Nombre d'erreurs : ", np.sum(kmeans.labels_ == train_labels))

        err.append(np.sum(kmeans.labels_ == train_labels) / taille * 20)
        inertie.append(np.round(kmeans.inertia_, 3))

    return err, inertie






