.. SCIFYsim documentation master file, created by
   sphinx-quickstart on Tue Jan 11 18:27:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SCIFYsim's documentation!
====================================

Quick start
-----------

After you have :ref:`installed <setup>` SCIFYsim, you can get taste of SCIFYsim by running the notebook ``SCIFYsim/scifysim/quick_start.ipynb``. This will get you through the initialization of a simulator object and make your first exposures.

What you need to know
---------------------


SCIFYsim does **not** have a GUI. You will have to interface with it through the API. You our recommendation is to use **jupyter(lab)** to create **semi-interactive scripts** to run the simulations and plot the data. You can also use **scripts** for a more rigid usage.

It uses:

* Possibly **extended stars** and **sources of interest** using *blackbody spectrum* or *provided spectra*.
* The **projection** of the observing array of pupils
* Absorption and thermal emission of the train of **optics** and **atmosphere**
* Random **optical path aberrations** from *fringe tracking* error power spectra
* Random **injection aberrations** based on partially *AO-corrected wavefront*
* *Atmospheric dispersion effects* and their *correction devices* (*Work in progress*)
* Beam combination in an **interchangeable combiner matrix** (including *chromatic aberrations*)
* Spectrograph and detector *integration*

It produces:

* Direct and differential (kernel) maps of the transmission in the field of view
* **Temporal series** of simulated observations
* Characterization of noises from

	+ **Thermal** background
	+ **Photon noise**
	+ **Instrumental aberrations**
	
* Computation of **detection sensitivity** based on advanced *hypothesis testing* techniques [1]_
  combining the evolution of the baselines and all the spectral channels.


:ref:`More information here. <need-to-know>`

.. toctree::
   :caption: Table of Contents
   :maxdepth: 2
   :glob:
   
   setup_guide
   need_to_know
   recipe
   modules
   scifysim.*

Other links
------------------

* :ref:`setup`
* :ref:`need-to-know`
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Building this documentation
---------------------------

Documentation is generated through sphinx. Intialization sequence follows this :ref:`recipe <docrecipe>`.

References:
+++++++++++

.. [1] Ceau et al. (2019), A&A, 630, A120