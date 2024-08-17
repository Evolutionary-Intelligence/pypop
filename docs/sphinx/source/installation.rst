Installation
============

In order to install *pypop7*, it is **highly recommended** to use the `Python3 <https://docs.python.org/3/>`_-based
virtual environment via `venv <https://docs.python.org/3/library/venv.html>`_ or
`conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_. Among them, `Anaconda <https://docs.anaconda.com/>`_
(or its mini version `miniconda <https://docs.conda.io/projects/miniconda/en/latest/>`_) is a very popular `Python
<https://www.python.org/>`_ programming platform (IDE) of scientists and engineers especially for Artificial Intelligence
(AI), Machine Learning (ML), Data Science, and Scientific Computing.

For **Virtual Environments**, please refer to `this online documentation
<https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>`_ for details.
In most cases, using virtual environments seems to be a good practice for `Python <https://www.python.org/>`_ projects.

Pip via Python Package Index (PyPI)
-----------------------------------

.. note:: The official website of PyPop7's Python source code is freely available at GitHub:
   https://github.com/Evolutionary-Intelligence/pypop.

Note that `pip <https://pip.pypa.io/en/stable/>`_ is the package installer for Python. You can use it to install
various open-source packages easily. For `pypop7`, please run the following **shell** command:

.. code-block:: bash

    pip install pypop7

For Chinese users, sometimes the following PyPI configuration can be used to speedup the installation process
of `pypop7` owing to possible network blocking:

.. code-block:: bash

    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
    pip config set install.trusted-host mirrors.aliyun.com

rather than the default PyPI setting:

.. code-block:: bash

    pip config set global.index-url https://pypi.org/simple
    pip config set install.trusted-host files.pythonhosted.org

(Note that other mirrors for PyPI could be also used here.)

If the latest cutting-edge version is preferred for development, you can install directly from the GitHub
repository of the increasingly popular `pypop7` library:

.. code-block:: bash
   
   git clone https://github.com/Evolutionary-Intelligence/pypop.git
   cd pypop
   pip install -e .

Conda-based Virtual Environment
-------------------------------

You can first use the popular `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_
(`Miniconda <https://docs.anaconda.com/miniconda/>`_) tool to create a virtual environment (e.g.,
named as `env_pypop7`):

.. code-block:: bash

    conda deactivate  # close exiting virtual env, if exists
    conda create -y --prefix env_pypop7  # free to change name of virtual env
    conda activate ./env_pypop7  # on Windows OS
    conda activate env_pypop7/  # on Linux
    conda activate env_pypop7  # on MacOS
    conda install -y --prefix env_pypop7 python=3.8.12  # create new virtual env
    pip install pypop7
    conda deactivate  # close current virtual env `env_pypop7`

Note that the above Python version (`3.8.12`) can be freely changed to meet your personal
**Python-3** version (>=3.5 if possible).

Although we strongly recommend to use the the `conda` package manager to build the virtual
environment as your working space, currently we do not add this library to `conda-forge
<https://conda-forge.org/>`_ and leave it for the future (maybe 2025). As a result,
currently you can only use `pip install pypop7` for `conda`.

For MATLAB Users
----------------

For MATLAB users, `MATLAB-to-Python Migration Guide
<https://www.enthought.com/wp-content/uploads/2019/08/Enthought-MATLAB-to-Python-White-Paper_.pdf>`_
or `NumPy for MATLAB Users <https://numpy.org/devdocs/user/numpy-for-matlab-users.html>`_ is highly
recommended. Given the fact that the USA government `blocks
<https://www.quora.com/Did-the-US-really-block-the-license-of-MATLAB-to-several-Chinese-universities>`_
the MATLAB license to several Chinese universities (including *HIT*, the affiliation of one core
developer), we argue that an increasing number of well-designed open-source software like Python,
NumPy, SciPy, and scikit-learn (just to name a few) are really wonderful alternatives to commercial
MATLAB in many cases.

For R Users
-----------

For R (and S-Plus) users, `NumPy-for-R <https://mathesaurus.sourceforge.net/r-numpy.html>`_
is highly recommended. Note that `R <https://www.r-project.org/>`_ is a free and well-established
software environment for statistical computing and graphics.

Uninstallation
--------------

If necessary, you could uninstall this open-source Python library *freely* with only one shell
command:

.. code-block:: bash

    pip uninstall -y pypop7

After you have installed it successfully, we wish that you could enjoy a happy journey on
**PyPop7** for black-box optimization.

.. image:: https://visitor-badge.laobi.icu/badge?page_id=Evolutionary-Intelligence.pypop
   :target: https://visitor-badge.laobi.icu/badge?page_id=Evolutionary-Intelligence.pypop
