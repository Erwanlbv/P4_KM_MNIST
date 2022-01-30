import numpy as np


""" Pour la partie avec un pré-traitement supplémentaire, voir class_lect_lign colonne pour l'utilisation"""


def lect_lignes(image):
    return np.tile(np.sum(image, axis=1)/len(image), len(image)).reshape(len(image), len(image[0])).transpose()


def lect_colonnes(image):
    return np.tile(np.sum(image, axis=0)/len(image[0]), len(image[0])).reshape(len(image), len(image[0]))


def lect_lig_et_col(image):
    return lect_lignes(image) + lect_colonnes(image) / 2


""" Zone des pré-traitements classiques, utilisés dans toutes les études de ce TP """


def lissage(image, seuil):
    return image * (image >= seuil)


def crop(dataset, row, column):
    L = []
    for i in range(len(dataset)):
        L.append(dataset[i, row:len(dataset[i])-row, column:len(dataset[i, 0])-column])
        dataset[i]
    return np.array(L)


def pre_pro(dataset, norm_bool, liss_bool, crop_bool):
    seuil = 200
    row_crop, col_crop = 3, 3

    if liss_bool:
        dataset = lissage(dataset, seuil)

    if norm_bool:
        dataset = dataset / 255

    if crop_bool:
        dataset = crop(dataset, row_crop, col_crop)

    return dataset




