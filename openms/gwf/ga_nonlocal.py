#
# @ 2023. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
# material to reproduce, prepare derivative works, distribute copies to the
# public, perform publicly and display publicly, and to permit others to do so.
#
# Author: Yu Zhang <zhy@lanl.gov>
#


from openms.gwf import ga_local

class GASCF(ga_local.GASCF):
    r"""Extended GA method for non-local correlation

    Brief introduction to the theoretical background:

    The Hamiltonain is

    .. math::
        H = \sum_I H^{loc}_I + \sum_{I\neq J} H^{loc}_{IJ}

    In general, the non-local correlation can contains three-site and four-site
    correlations. Here, we only consider the two-site correlation.

    And we assume the following formula for the two-site correlaiton:

    .. math::
        H_{IJ} = \sum_{pqrs} J_{IJ} n_{I, pq} n_{J, rs}

    which is dipole-dipole-like two-site interaction.


    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
