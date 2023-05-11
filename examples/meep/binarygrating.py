#!
import os
import inspect
import math

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

import meep as mp

resolution = 25  # pixels/μm

dpml = 1.0  # PML thickness
dsub = 3.0  # substrate thickness
dpad = 3.0  # padding between grating and PML
gp = 10.0  # grating period
gh = 0.5  # grating height
gdc = 0.5  # grating duty cycle

nperiods = 10  # number of unit cells in finite periodic grating

ff_distance = 1e8  # far-field distance from near-field monitor
ff_angle = 20  # far-field cone angle
ff_npts = 500  # number of far-field points

ff_length = ff_distance * math.tan(math.radians(ff_angle))
ff_res = ff_npts / ff_length

sx = dpml + dsub + gh + dpad + dpml
cell_size = mp.Vector3(sx)

pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

symmetries = [mp.Mirror(mp.Y)]

wvl_min = 0.4  # min wavelength
wvl_max = 0.6  # max wavelength
fmin = 1 / wvl_max  # min frequency
fmax = 1 / wvl_min  # max frequency
fcen = 0.5 * (fmin + fmax)  # center frequency
df = fmax - fmin  # frequency width

src_pt = mp.Vector3(-0.5 * sx + dpml + 0.5 * dsub)
sources = [
    mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=src_pt)
]

k_point = mp.Vector3()


