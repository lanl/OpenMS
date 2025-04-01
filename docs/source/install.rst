.. _installation:

Installation
************

.. _compile_libraries and c/c++/fortran_extensions:

Manual installation from the Github repo
========================================

Manual installation requires `cmake <http://www.cmake.org>`_,
`numpy <http://www.numpy.org/>`_, `scipy <http://www.scipy.org/>`_,
`MEEP <https://github.com/NanoComp/meep>`_ (optional), and
`TiledArray <https://github.com/ValeevGroup/tiledarray>`_ (optional).

You can download the latest OpenMS (or the development branch) from GitHub::

 $ git clone https://github.com/lanl/OpenMS
 $ cd openms
 $ git checkout develop # optional if you'd like to try out the development branch

Build the libs and other extensions in :file:`openms/lib`::

  $ cd openms/lib
  $ bash build.sh

Alternatively::

  $ cd openms/lib
  $ cd build
  $ cmake -DCMAKEflags ../
  $ make

This will automatically download required libs and compile them.
Finally, to make Python find the :code:`openms` package, add the top-level :code:`openms` directory (not
the :code:`openms/openms` subdirectory) to :code:`PYTHONPATH`. For example::

 export PYTHONPATH=/path/to/openms:$PYTHONPATH

To ensure the installation is successful, start a Python shell, and type::

 >>> import openms

.. If Meep is installed, it's also required to make Python to find the :code:`meep` package, which is installed
.. in the `/path/to/top_openms/openms/lib/deps/lib/python{version}/site-packages/`::
..
..  export PYTHONPATH=/path/to/top_openms/openms/lib/deps/lib/python{version}/site-packages/:$PYTHONPATH


cmake configurations
--------------------
The `CMAKEflags` should be replaced with your proper cmake options, such as `-DCMAKE_PREFIX_PATH`,
List of available options:

* `ENABLE_MPI` -- Set to `ON` to turn on MPI support [Default OFF]

* `ENABLE_MEEP` -- Set to `ON` to install MEEP FDTD solver for Maxwell's equations [Default OFF]

* `ENABLE_TA` -- Set to `ON` to install TiledArray for tensor contraction[Default OFF]

* `ENABLE_TACUDA` -- Set to `ON` to turn on Cuda GPU support in TA lib [Default OFF]

* `toolchainpath` -- Set toolchain path to for compiling TA [Default cmake/vg/toolchains/]

* `CMAKE_BUILD_TYPE` -- Set build type [Default `Release`]

* `BUILD_INFDTD` -- Set `ON` to build internal FDTD solver [Default `ON`]

more details TBA
