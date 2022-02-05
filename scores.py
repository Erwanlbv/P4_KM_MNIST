import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score


def compute_scores(dataset, km, inertia_bool, sil_bool, davies_bool):
    output = [None, None, None]

    if inertia_bool:
        output[0] = np.round(km.inertia_, 3)

    if sil_bool: # On cherche à la maximiser, prends des valeurs entre -1 et 1
        output[1] = np.round(silhouette_score(dataset, km.labels_), 3)

    if davies_bool: # On cherche à le minimiser (la valeur minimale étant 0)
        output[2] = np.round(davies_bouldin_score(dataset, km.labels_), 3)

    return output


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


