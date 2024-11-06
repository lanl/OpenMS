QED mean-field calculation
--------------------------

QEDHF with coherent state
^^^^^^^^^^^^^^^^^^^^^^^^^

The basic steps of setting up a QEDHF calculation is as follows

1. Set up a molecule from Pyscf:

>>> import numpy
>>> from pyscf import gto
>>> mol         = gto.Mole()
>>> mol.atom  = f"C     0.00000000     0.00000000    0.0;\
>>>               O     0.00000000     1.23456800    0.0;\
>>>               H     0.97075033    -0.54577032    0.0;\
>>>               C    -1.21509881    -0.80991169    0.0;\
>>>               H    -1.15288176    -1.89931439    0.0;\
>>>               C    -2.43440063    -0.19144555    0.0;\
>>>               H    -3.37262777    -0.75937214    0.0;\
>>>               O    -2.62194056     1.12501165    0.0;\
>>>               H    -1.71446384     1.51627790    0.0"
>>> mol.verbose = 3
>>> mol.basis = 'STO-3G',

2. set up cavity properties (polarization vector and frequencies)

For the light-matter interaction, the cavity properties need to be specified,
including number of modes, frequencies and the polarization vector of each mode:

>>> nmode = 1
>>> cavity_freq = numpy.zeros(nmode)
>>> cavity_mode = numpy.zeros((nmode, 3))
>>> cavity_freq[0] = 3.0 /27.211386245988
>>> cavity_mode[0,:] = 0.1 * numpy.asarray([1, 1, 1])

3. create the QEDHF mean-field object in the same maner as pyscf:

>>> from openms.openms import qedhf
>>>
>>> qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
>>> qedmf.max_cycle = 500

4. fianlly, perform the calculation

>>> qedmf.kernel()

QEDHF with fock state
^^^^^^^^^^^^^^^^^^^^^

Add "z_alpha" variable into the QEDHF class will force the calculation in Fock
state representation:

>>> from openms.openms import qedhf
>>>
>>> qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode,
>>>         z_alpha = numpy.zeros(nmode),
>>>         cavity_freq=cavity_freq)
>>> qedmf.max_cycle = 500

Alternatively, you can mannually set the "use_cs" to False:

>>>
>>> qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode,
>>>         cavity_freq=cavity_freq)
>>> qedmf.use_cs = False


SC-QEDHF calculation
^^^^^^^^^^^^^^^^^^^^

>>> from openms.openms import scqedhf
>>>
>>> qedmf = scqedhf.RHF(mol, xc=None, cavity_mode=cavity_mode,
>>>         cavity_freq=cavity_freq)
>>> qedmf.max_cycle = 500
>>> qedhf.kernel()


VT-QEDHF calculation
^^^^^^^^^^^^^^^^^^^^


>>> from openms.openms import vtqedhf
>>>
>>> qedmf = vtqedhf.RHF(mol, xc=None, cavity_mode=cavity_mode,
>>>         cavity_freq=cavity_freq)
>>> qedmf.max_cycle = 500
>>> qedhf.kernel()


VSQ-QEDHF calculation
^^^^^^^^^^^^^^^^^^^^^

This is very similar to the VT-QEDHF calculations, only need to turn
on the "optimize_vsq":

>>> from openms.openms import vtqedhf
>>>
>>> qedmf = vtqedhf.RHF(mol, xc=None, cavity_mode=cavity_mode,
>>>         cavity_freq=cavity_freq)
>>> qedmf.max_cycle = 500
>>> qedmf.qed.optimize_vsq = True
>>> qedhf.kernel()
