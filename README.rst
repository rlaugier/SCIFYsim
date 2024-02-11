SCIFYsim: a python package for interferometric nulling simulation
=================================================================

It is a python module designed to simulate the behaviour of high-contrast interferometric instruments, with a focus on the Very Large Telescope Interferometer. It can also be easily applied to other ground-based and space facilities.

SCIFYsim aims at replacing GENIEsim [1]_ for all the simulations in the SCIFY project. It is specifically developped to accompany the development and implementation of future generations of advanced nulling techniques [2]_ [3]_.

Principles:
-----------

SCIFYsim uses:

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
	
* Computation of **detection sensitivity** based on advanced *hypothesis testing* techniques [4]_
  combining the evolution of the baselines and all the spectral channels.

Documentation:
--------------

Find growing `API documentation here <https://rlaugier.github.io/scifysim_doc.github.io>`_


Acknowledgement
---------------

SCIFYsim is a development carried out in the context of the `SCIFY project <http://denis-defrere.com/scify.php>`_. `SCIFY <http://denis-defrere.com/scify.php>`_
has received funding from the **European Research Council (ERC)** under the
European Union's Horizon 2020 research and innovation program (*grant agreement No 866070*).  


Dependencies:
-------------

Last tested with python 3.9 . The following packages are required:

- kernuller `<https://github.com/rlaugier/kernuller>`_
- xaosim `<https://github.com/fmartinache/xaosim>`

Optional:

- dask *To allow the construction of larger sensitivity maps*
- jupyterlab *To interact with the similator in enriched interactive scripts*

Recommandation for installation:
--------------------------------

.. code-block::

 	python setup.py install
 
See the dedicated `installation page <https://rlaugier.github.io/scifysim_doc.github.io/setup_guide.html#setup>`_ of the documentation.


Rerences:
+++++++++
.. [1] Absil et al. (2006), A&A, 448, 787-800.
.. [2] Martinache & Ireland (2018), A&A, 619, A87
.. [3] Laugier et al. (2020), A&A, 642, A202
.. [4] Ceau et al. (2019), A&A, 630, A120