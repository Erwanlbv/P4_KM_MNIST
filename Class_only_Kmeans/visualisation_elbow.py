import matplotlib.pyplot as plt
import numpy as np

import joblib as jb
from class_kmeans_only import kmeans_only
from variables import *


taille = 60000
err, inertie = kmeans_only(array_train_dataset, array_train_labels, taille, 5, 15)

#if err < 1.5:
#    jb.dump(kmeans, 'km_only_' + str(err_pourc) + '_' + str(i) + '.joblib')


fig, axs = plt.subplots(2, 1, figsize=(13, 7))
fig.suptitle('Km only \n taille : ' + str(taille))

axs[0].set_title(" Pourcentage d'erreur (dataset d'entrainement")
axs[0].plot(range(5, 15), err)

axs[1].set_title(" Inertie du modèle ")
axs[1].plot(range(5, 15), inertie)

for ax in axs.flat:
    ax.legend()

    plt.show()
    plt.pause(0.01)