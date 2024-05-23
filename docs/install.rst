============
Installation
============

Prerequisites
-------------

To use QCFPGA you will need to be using `Python 3.6 or later <https://www.python.org/downloads/>`_.
You will also need to ensure that you have an `OpenCL <https://www.khronos.org/opencl/>`_ implementation installed. 
This is done by default on MacOS, but you shuld check that ``clinfo`` or some other diagnostic command will run.


Installing from PyPI
--------------------

This library is distributed on `PyPI <https://pypi.python.org/pypi/qcfpga>`_ and can be installed using pip:

.. code:: sh

   $ pip install qcfpga

If you run into any issues, you should try installing from source.

Installing from Source
----------------------

You can install QCFPGA from the source. First, clone the repository off
GitHub:

.. code:: sh

   $ git clone https://github.com/qcfpga/qcfpga

Then you will need to ``cd`` into the directory, and install the
requirements.

.. code:: sh

   $ pip install -r requirements.txt

And finally you can install:

.. code:: sh

   $ python setup.py install
