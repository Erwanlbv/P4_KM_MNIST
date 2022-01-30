import numpy as np
import matplotlib.pyplot as plt
from variables import *
from pre_proc_programs import *

#Pour observer les 10 premiers éléments du dataset
"""for i in range(10):
    print(array_train_dataset[i])
    plt.imshow(array_train_dataset[i], cmap='gray')
    plt.show()
"""

#Pour observer les 5 premiers élements du dataset, avec un réduction puis avec réduction et lissage
"""for i in range(5):
    im = array_train_dataset[i]

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(im, cmap='gray')
    axs[1].imshow(crop(im, 3, 3), cmap='gray')
    axs[2].imshow(lissage(crop(im, 3, 3), 200), cmap='gray')

    fig.show()
"""

#Pour observer les 5 premiers éléments du dataset dont la classe est définie par nb
i = 0
nb = 7
count = 0
labels = []
while count < 5:
    label = array_train_labels[i]
    print(label)
    if label == nb:
        labels.append(i)
        count += 1
    i += 1

fig, axs = plt.subplots(len(labels), 2)

for id, label in enumerate(labels):
    axs[id, 0].imshow(array_train_dataset[label], cmap='gray')
    axs[id, 1].imshow(lect_lig_et_col(array_train_dataset[label]), cmap='gray')

fig.show()