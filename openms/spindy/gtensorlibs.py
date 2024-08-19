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

import sys
import numpy


def string_to_numpy_array(matrix_string):
    # Split the string into lines
    lines = matrix_string.strip().split('\n')

    # Split each line into elements and convert to float
    matrix_data = [list(map(float, line.split())) for line in lines]

    # Convert the list of lists to a NumPy array
    return numpy.array(matrix_data)


gtensor_data = {}
gtensor_data['sample'] = numpy.array([[2., 0., 0.],
                                  [0., 2., 0.],
                                  [0., 0., 2.]])

#------------------------
# from ORCA calculations:
#------------------------

# VOPcOH8 default I=3.5
g = f"""
              1.9813375    0.0039050    0.0021903
              0.0038804    1.9891492   -0.0008947
              0.0021890   -0.0009015    1.9902479
"""
gtensor_data['VOPcOH8'] = string_to_numpy_array(g)

# CuPcOH8  # default I=1.5
g = f"""
              2.0730877   -0.0171320   -0.0096586
             -0.0171445    2.0349693    0.0036730
             -0.0096388    0.0037221    2.0304833
"""
gtensor_data['CuPcOH8'] = string_to_numpy_array(g)

# CoPcOH8 default I = 3.5
g = f"""
              2.0559245    0.1082167    0.0609306
              0.1081040    2.2957483   -0.0233811
              0.0608570   -0.0234408    2.3241520
"""


if __name__ == "__main__":
    for molecule in ["sample", "CoPcOH8", "CuPcOH8", 'VOPcOH8']:
        print(f"\n gtensor_data[{molecule}]=\n{gtensor_data[molecule]}")
