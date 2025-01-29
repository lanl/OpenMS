
import numpy

def get_mean_std(energies, ratio=10):
    m = max(1, len(energies) // ratio)
    last_m_real = numpy.asarray(energies[-m:]).real
    mean = numpy.mean(last_m_real)
    std_dev = numpy.std(last_m_real)
    return mean, std_dev

