.. _setup:

Installing SCIFYsim
===================

Prerequisites
-------------

Managing your environments through ``conda`` is highly recommended.

Kernuller
---------

You can obtain ``kernuller`` from its `github page <https://github.com/rlaugier/kernuller>`_. 

.. code-block::
	
	git clone https://github.com/rlaugier/kernuller
	cd kernuller
	python setup.py install
    
XAOSIM
-------

You can obtain ``xaosim`` from its `github page <https://github.com/fmartinache/xaosim>`_

.. code-block::
    
    git clone https://github.com/fmartinache/xaosim
    cd xaosim
    python setup.py install

.. warning::
    
    The xaosim fails to install on windows, but the part that is problematic is
    not used in SCIFYsim. If you encounter a problem here, open the ``setup.py`` file,
    remove the line that stars with ``data_files`` and try again.

Installing SCIFYsim
-------------------

Installation procedure is similar:

.. code-block::
	
	git clone https://github.com/rlaugier/scifysim
	cd SCIFYsim
	python setup.py install

Setting up a workspace
----------------------
Recommended workspace setup:
::

	└── working_directory
		  ├── local_config
		  │   ├── default_R200.ini
		  │   └── vega_R200.ini
		  ├── log
		  └── my_work.ipynb
		  
Log directory
+++++++++++++

SCIFYsim uses a automated logging system. It will require a **log directory** in your working directory.


Config files
++++++++++++

SCIFYsim uses **config files**. You can get started by copying one of the default config files in ``SCIFYsim/scifysim/config`` into your working directory. You will have to refer to it explicitly in your source code.

Config files are standard ``.ini`` files but use comments with ``#``. They have sections marked by [] and the parameters use keys. The different organs of the simulator often keep in ``self.config`` the reference to the parsed config file that was used to create them. You can access their content by 

* ``self.config.get("section", "key")`` for **strings**
* ``self.config.getint("section", "key")`` for integers
* ``self.config.get_array("section", "key")`` for arrays
* ``self.config.getboolean("section", "key")`` for booleans



.. admonition:: Photometry reference

	SCIFYsim uses references to Vega to compute photometry in a relevant manner. Therefore you need a to maintain a vega config file up to date that uses the same instrumental configuration.

