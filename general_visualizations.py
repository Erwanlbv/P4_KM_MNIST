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


def get_label_real_class(true_labels, pred_labels):
    all_labels = np.unique(pred_labels)
    label_equivalence = []

    for label in all_labels:
        index_classes = np.where(pred_labels == label)                                    # On obtient tous les indices où km_labels_ vaut label
        labels_in_true_labels = np.unique(true_labels[index_classes], return_counts=True) # On regarde la valeur des vrais labels en chacun de ces indices
        res = labels_in_true_labels[0][np.argmax(labels_in_true_labels[1])]                  # On choisit le label avec le plus gros nombre do'ccurences.
        label_equivalence.append(res)

    #print('Liste des équivalences', label_equivalence) # label_equivalence[0] correspond à la classe 'réelle' du 1er regroup. [1] du 2e..
    return np.array(label_equivalence)


def compute_error(true_labels, pred_labels):

    label_equivalence = get_label_real_class(true_labels, pred_labels)
    error_per_class = []

    for i in range(10):
        res_matrix = np.zeros(true_labels.shape)
        indexs = np.where(label_equivalence == i)[0]
        #print('Indices de la classe ' + str(i) + ' :', indexs)

        for index in indexs:
            res_matrix += (pred_labels == index)

        error_per_class.append(np.sum(res_matrix != (true_labels == i)))

    return error_per_class




