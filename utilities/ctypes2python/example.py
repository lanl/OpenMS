import numpy as np
import ctypes

# Load the shared library
lib = ctypes.CDLL('./libexample.so')

# Define the argument types for the C function
lib.array_double.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='CONTIGUOUS'), ctypes.c_int]

# Create a NumPy array
arr = np.array([1.2, 2, 3, 4, 5], dtype=np.float64)

# Call the C function with the NumPy array
lib.array_double(arr, arr.size)

# The original NumPy array is modified in place
print(arr)  # Output: [ 2  4  6  8 10]

