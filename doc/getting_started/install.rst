Installation
************

Renate is available via PyPI and can be installed using :code:`pip`:

.. code-block:: bash

    pip install Renate

If you want to use additional methods that require the Avalanche library, please use

.. code-block:: bash

    pip install Renate[avalanche]

If you want to use Renate for :doc:`benchmarking <../benchmarking/index>`, please use

.. code-block:: bash

    pip install Renate[benchmark]

Renate contributors should use

.. code-block:: bash

    pip install Renate[dev]

This will install further dependencies which are required for code formatting and unit testing.

Alternatively, if you want to access the code directly (e.g., for developing and running new methods)
it is possible to clone the git repository

.. code-block:: bash

    git clone https://github.com/awslabs/Renate.git

We also recommend using a virtual environment to avoid conflicts in the 
versions of the packages installed. For example, following
`these instructions <https://docs.python.org/3/library/venv.html>`_ to use :code:`venv`.
