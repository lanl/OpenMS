# myclass.py

import _my_module
import mpi_example


class MyClass(object):
    def __init__(self):
        self.g = [[1,2,3], [4,5,6], [7,8,9]]

    def cpp_method(self):
        g2 = _my_module.cpp_method_impl(self)
        #self.g = _my_module.cpp_method_impl(self.g)
        print(g2)
        print(self.g)


if __name__ == "__main__":
    obj = MyClass()
    obj.cpp_method()

    mpi_example.initialize_mpi()

    result = mpi_example.mpi_sum()
    if mpi_example.get_mpi_rank() == 0:
        print("MPI sum result:", result)

    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = mpi_example.parallel_sum(numbers)
    if mpi_example.get_mpi_rank() == 0:
        print("Sum of numbers:", result)

    mpi_example.finalize_mpi()

    #
    obj = _my_module.MyClass("Test", 42)
    print(obj.getName())
    obj.setValue(100)
    print(obj.getValue())

