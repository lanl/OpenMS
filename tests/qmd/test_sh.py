import unittest
import numpy as np
from pyscf import gto
from openms.models import tully


def plot_e(x, energies):
    import matplotlib.pyplot as plt
    # plot
    nm = energies.shape[0]

    # fig, ax = plt.subplots(nm, 1, sharex=True, wspace=0)
    fig = plt.figure(figsize=(5, 2.5*nm))
    gs = fig.add_gridspec(nm, hspace=0)
    ax = gs.subplots(sharex=True)

    if nm == 1: ax = [ax]

    for im in range(nm):
        ax[im].plot(x, energies[im, :, 0], linewidth=2, color="black")
        ax[im].plot(x, energies[im, :, 1], linewidth=2, color="red")

        ymin = np.min(energies[im]) * 1.2
        ymax = np.max(energies[im]) * 1.2

        ax[im].set_ylim(ymin, ymax)
    #plt.show()
    plt.savefig("tullymodel.pdf")

class TestTully(unittest.TestCase):
    def test_tullymodels(self):

        coords = np.arange(-10, 10, 0.2)
        refs = np.zeros((3, len(coords), 2))

        mol = gto.M(verbose=3)
        models = [tully.TullyModel1, tully.TullyModel2, tully.TullyModel3]

        energies = np.zeros((len(models), len(coords), 2))

        for im, model in enumerate(models):
            tm1 = model(mol)
            for i, x in enumerate(coords):
               tm1.get_H(x)
               tm1.diag()
               energies[im, i] = tm1.e
        print("energies:\n",energies)
        # plot_e(coords, energies)

if __name__ == '__main__':
    unittest.main()
