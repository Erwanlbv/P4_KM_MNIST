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





