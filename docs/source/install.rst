.. _installing:

Installation
************

Installation with pip
=====================

This is the recommended way to install OpenMS::

 $ pip install openms (TBA)

.. _compile_libraries and c/c++/fortran_extensions:

Manual installation from the Gthub repo
=======================================

Manual installation requires `cmake <http://www.cmake.org>`_,
`numpy <http://www.numpy.org/>`_, `scipy <http://www.scipy.org/>`_,
`meep <https://github.com/NanoComp/meep>`_ (optional), and 
`TiledArray <https://github.com/ValeevGroup/tiledarray>`_ (optional).

You can download the latest OpenMS (or the development branch) from GitHub::

 $ git clone https://github.com/lanl/OpenMS
 $ cd openms
 $ git checkout develop # optional if you'd like to try out the development branch

Build the libs and other extensions in :file:`openms/lib`::

 $ cd openms/lib
 $ mkdir build
 $ cd build
 $ cmake ..
 $ make

This will automatically download required libs and compile them.
Finally, to make Python find the :code:`openms` package, add the top-level :code:`openms` directory (not
the :code:`openms/openms` subdirectory) to :code:`PYTHONPATH`. For example::

 export PYTHONPATH=/path/to/openms:$PYTHONPATH

To ensure the installation is successful, start a Python shell, and type::

 >>> import openms

more details TBA

