from class_k_only import km_only
from scores import compute_scores, compute_error
from variables import *

TAILLE = [500, 5000, 10000]
MIN_CLUSTERS = 2
MAX_CLUSTERS = 40


def visualisation_km_only():

    norm_bool, liss_bool, crop_bool = True, True, True

    fig, axs = plt.subplots(3, figsize=(13, 7))
    fig.suptitle("Kmeans only " +
                 '\n Normalisation : ' + str(norm_bool) +
                 ' Lissage : ' + str(liss_bool) +
                 ' Réduction : ' + str(crop_bool), fontsize=15)

    for taille in TAILLE:
        res = np.zeros((MAX_CLUSTERS - MIN_CLUSTERS + 1, 3))
        dataset = np.copy(array_train_dataset[:taille])

        for i in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
            print('Début des calculs pour ' + str(i) + ' clusters')
            _, km = km_only(np.copy(dataset), i, norm_bool=norm_bool, liss_bool=liss_bool, crop_bool=crop_bool)
            res[i - MIN_CLUSTERS] = compute_scores(_, km, norm_bool, liss_bool, crop_bool)
            print('Résultats :' + str(res[i - MIN_CLUSTERS]))

        for id, ax in enumerate(axs.flat):
            ax.set_title("Inertie du modèle : " * (id == 0) + 'Score Silouhette : ' * (id == 1) + 'Davies Score' * (id == 2))
            ax.plot(range(MIN_CLUSTERS, MAX_CLUSTERS + 1), res[:, id], label=(str(taille) + ' images'))
            if id < 2:
                ax.get_xaxis().set_visible(False)
                ax.set_xlabel('Nombre de clusters')
            ax.legend()

    fig.show()


def small_visu_km_only():
    norm_bool, liss_bool, crop_bool = True, True, True

    fig, axs = plt.subplots(2, figsize=(13, 7))
    fig.suptitle("Kmeans only " +
                 '\n Normalisation : ' + str(norm_bool) +
                 ' Lissage : ' + str(liss_bool) +
                 ' Réduction : ' + str(crop_bool), fontsize=15)

    for taille in TAILLE:
        dataset = np.copy(array_train_dataset[:taille])
        true_labels = array_train_labels[:taille]

        res = np.zeros((MAX_CLUSTERS - MIN_CLUSTERS + 1, 3))
        erreur = np.zeros(MAX_CLUSTERS + 1 - MIN_CLUSTERS)

        for i in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
            print('Début des calculs pour ' + str(i) + ' clusters')

            _, km = km_only(np.copy(dataset), i, norm_bool=norm_bool, liss_bool=liss_bool, crop_bool=crop_bool)
            res[i - MIN_CLUSTERS] = compute_scores(_, km, norm_bool, liss_bool, crop_bool)
            erreur[i - MIN_CLUSTERS] = np.sum(compute_error(true_labels, km.labels_))/(2*taille) * 100

            print('Résultats :' + str(res[i - MIN_CLUSTERS]))

        for id, ax in enumerate(axs.flat):
            ax.set_title((id == 0)*'Inertie du modèle' + (id == 1)*'Pourcentage de mauvaises classifications')
            ax.plot(res[:, 0] * (id == 0) + erreur * (id == 1), label=(str(taille) + ' images'))
            ax.legend()

    fig.show()


#visualisation_km_only()
#small_visu_km_only()