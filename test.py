import numpy as np
import matplotlib.pyplot as plt
from variables import *
from pre_proc_programs import *


"""for i in range(10):
    print(array_train_dataset[i])
    plt.imshow(array_train_dataset[i], cmap='gray')
    plt.show()
"""

for i in range(5):
    im = array_train_dataset[i]

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(im, cmap='gray')
    axs[1].imshow(crop(im, 3, 3), cmap='gray')
    axs[2].imshow(lissage(crop(im, 3, 3), 200), cmap='gray')

    fig.show()
