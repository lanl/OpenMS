import unittest
from openms.lib.backend import backend as  bd
from openms.lib.backend import set_backend, TABackend
import numpy

def make_test_case(backend, N=10, order=4):
  
  set_backend(backend)
  if isinstance(bd, TABackend):
      bd.blksize = N// 2
      print("blksize=", bd.blksize)

  class TestCase(unittest.TestCase):

    def test_einsum(self):
      print('backend is:', backend)
      order = 4
      dims = [N + i for i in range(order)]
      
      print("test einsum")
      print('dimensions are:', *dims)

      A = bd.rand(dims)
      print("create tensor B")
      B = bd.rand(dims[2:])
      C = bd.einsum("ijkl,kl->ij", A, B)

      print('C=\n', numpy.array(C))
  return TestCase

N = 10
class ArrayTest(make_test_case("numpy", N)): pass

try:
    import torch
    set_backend("torch")
    class TorchArrayTest(make_test_case(bd, N)): pass
except ImportError:
    print("torch is not available")

try:
    import tiledarray as ta
    print("TA is available")
    #class TAArrayTest(make_test_case("numpy", 50)): pass
    class TAArrayTest(make_test_case("TiledArray", N)): pass
except ImportError:
    print("TA is not available")

print("\n ======= start test ========\n")

if __name__ == '__main__':
  unittest.main()
