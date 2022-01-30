import torch
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import numpy as np
import matplotlib.pyplot as plt

#kmeans = sklearn.cluster.KMeans()

training_data = datasets.MNIST(
    root='/Users/erwan/PycharmProjects/P4_Ks_MNIST/data',
    train=True,
    download=True,
)

testing_data = datasets.MNIST(
    root='/Users/erwan/PycharmProjects/P4_Ks_MNIST/test_data',
    train=False,
    download=True,
)


array_train_dataset = np.copy(training_data.data.numpy())
array_train_labels = np.copy(training_data.targets.numpy())

array_test_dataset = np.copy(testing_data.data.numpy())
array_test_labels = np.copy(testing_data.targets.numpy())



def display():
    print('Dataset', dir(training_data)) #Chaque élément du dataset est de la forme (Image PIL, classe)
    print('Element of the dataset', training_data.data)
    print("Représentation de l'élément", training_data[0])
    print('Image to array', np.asarray(training_data[0][0]))
    print(len(training_data))

    #Deux manières d'afficher des éléments de la base de données :

    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(np.asarray(training_data[0][0]), cmap='gray')
    axs[0].set_title('Figure via np.asarray')

    axs[1].matshow(training_data[0][0], cmap='gray')
    axs[0].set_title('Figure via mathshow')

    #plt.show()
    plt.pause(1)













