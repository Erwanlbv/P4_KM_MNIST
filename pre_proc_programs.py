import numpy as np


def lissage(image, seuil):
    return image * (image >= seuil)


def lect_lignes(image):
    return np.tile(np.sum(image, axis=1)/len(image), len(image)).reshape(len(image), len(image[0])).transpose()


def lect_colonnes(image):
    return np.tile(np.sum(image, axis=0)/len(image[0]), len(image[0])).reshape(len(image), len(image[0]))


def lect_lig_et_col(image):
    return lect_lignes(image) + lect_colonnes(image) / 2




