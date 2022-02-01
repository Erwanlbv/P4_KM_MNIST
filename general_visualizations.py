import matplotlib.pyplot as plt
import numpy as np


def display_histos(pred_labels, true_labels):

    fig, axs = plt.subplots(2, figsize=(11, 7))
    fig.suptitle('Histogrammes réel et prédit, pour ' + str(len(true_labels)) + 'images', fontsize=15)

    axs[0].set_title('Histogramme réel')
    axs[0].hist(true_labels)

    axs[1].set_title('Histogramme prédit')
    axs[1].hist(pred_labels,
                range=(0, 10 * (len(np.unique(pred_labels)) < 10) + len(np.unique(pred_labels)) * (len(np.unique(pred_labels)) >= 10)),
                rwidth=0.5,
                )

    fig.show()


def get_histo(km, true_labels, n_clusters):

    for i in range(n_clusters):
        fig, ax = plt.subplots(3, fontsize=15)
        fig.suptitle('Histogramme du cluster n° ' + str(i))

        # Affiche le centre du cluster


def get_label_real_class(true_labels, pred_labels):
    all_labels = np.unique(pred_labels)
    index_classes = []
    label_equivalence = []

    for label in all_labels:
        index_classes.append(np.where(pred_labels == label))
        res = [np.sum(true_labels[index_classes] == i) for i in range(len(all_labels))]
        label_equivalence.append(res[np.argmax(res)])

    print('Liste des équivalences', label_equivalence) # label_equivalence[0] correspond à la classe 'réelle' du 1er regroup. [1] du 2e..
    return label_equivalence



