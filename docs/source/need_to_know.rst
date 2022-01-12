.. _need-to-know:

What you need to know
=====================

Philosophy
----------

SCIFYsim does not have a GUI. You will have to interface with it through the API. You our recommendation is to use **jupyter(lab)** to create **semi-interactive scripts** to run the simulations and plot the data. You can also use **scripts** for a more rigid usage.

Simulator
---------

One can use ``utilities.prepare_all`` as a shortcut to start a new simulator from a config file.

The simulator object is organized as follows:

- simulator

	* **source** "``src``": Computes and holds the parameters of the target
	* **injector** "``injector``"
	
		+ **atmo** "``screen``": list of wavefront screens
		+ **fiber_head** "``fiber``": Computes and holds the information
			of the waveguide for the injection
		+ **focuser** "``focal_plane``": The system to simulate injection
		
	* **corrector** "``corrector``": Models the adjustments we can make to
	  the independent beams:
	  
	  + Optical path
	  + Longituninal dispersion
	  + etc. 
	* **fringe_tracker** "``fringe_tracker``":
	* **observatory** "``obs``": Contains the information on the array and its projection depending on the direction of observation.
	* **combiner** "``combiner``": Computes and holds the combiner matrix
	* **spectrograph** "``spectro``":
	* **integrator** "``integrator``": Simulates the integrating behavior of light sensitive pixels.
	
Simulating an exposure
----------------------

First you must point the array to the target using the method ``simulator.point``. This updates the projection parameters and a few other things.

Call ``simulator.make_exposure`` to make record light. This returns an integrator object containing all the parameters of the exposure.

Useful tools
------------

You will find useful tools in the modules:

* scifysim.utilities
* scifysim.plot_tools
