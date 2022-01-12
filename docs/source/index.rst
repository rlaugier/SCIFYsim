.. SCIFYsim documentation master file, created by
   sphinx-quickstart on Tue Jan 11 18:27:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SCIFYsim's documentation!
====================================

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   setup

.. toctree::
   :caption: Table of Contents
   :maxdepth: 4
   :glob:

   scifysim.*

Quick start
-----------

After you have :ref:`installed <setup>` SCIFYsim, you can get taste of SCIFYsim by running the notebook ``SCIFYsim/scifysim/quick_start.ipynb``. This will get you through the initialization of a simulator object and make your first exposures.

What you need to know
---------------------

SCIFYsim does not have a GUI. You will have to interface with it through the API. You our recommendation is to use **jupyter(lab)** to create **semi-interactive scripts** to run the simulations and plot the data. You can also use **scripts** for a more rigid usage.

:ref:`More information here. <need-to-know>`



Indices and tables
------------------

* :ref:`setup`
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Building this documentation
---------------------------

Documentation is generated through sphinx. Intialization sequence follows this :ref:`recipe <docrecipe>`.

