from pre_proc_programs import *
from variables import *


for i in range(20):
    new_im = lect_lig_et_col(np.asarray(training_data[i][0]))

    plt.title('Classe n° ' + str(training_data[i][1]))
    plt.imshow(new_im, cmap='gray')
    plt.pause(0.01)


# On pourrait afficher une figure de chaque, avant et après pré-traitement, dans un tableau 10, 2
