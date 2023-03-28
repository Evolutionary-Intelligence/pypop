Installation of PyPop7
======================

In order to install *pypop7*, it is **highly recommended** to use the Python3-based virtual environment via
`venv <https://docs.python.org/3/library/venv.html>`_ or
`conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_:

Pip via Python Package Index (PyPI)
-----------------------------------

.. code-block:: bash

    pip install pypop7

Note that `pip <https://pip.pypa.io/en/stable/>`_ is the package installer for Python. You can use it to install
various packages easily from the Python Package Index (PyPI).

Conda-based Virtual Environment
-------------------------------

You can first use the very popular `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ tool to create
a virtual environment (e.g. named as `env_pypop7`):

.. code-block:: bash

    conda deactivate  # close exiting virtual env
    conda create -y --prefix env_pypop7  # you can change the name of virtual environment to any
    conda activate env_pypop7  # for Windows OS (use `conda activate env_pypop7/` for Linux)
    conda install -y --prefix env_pypop7 python=3.8.12
    pip install pypop7
    conda deactivate

Note that the above Python version (`3.8.12`) can be changed to meet your personal Python3 version (>=3.5).

Although we strongly recommend to use the the `conda` package manager to build the virtual environment as your working
space, currently we do not add this library to `conda-forge <https://conda-forge.org/>`_ and leave it for the future
(maybe 2023/2024). As a result, you can only use `pip install pypop7` for `conda`.
