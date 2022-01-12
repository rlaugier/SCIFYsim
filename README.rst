SCIFYsim: a python package for interferometric nulling simulation
=================================================================

It is a python module designed to simulate the behaviour of high-contrast interferometric instruments, both on the ground and in space.
SCIFYsim aims at replacing GENIEsim for all the simulations in the SCIFY project.

- Absil et al. (2006), A&A, 448, 787-800.

Documentation:
--------------

Find growing API documentation `here <https://rlaugier.github.io/scifysim_doc.github.io>`_


Acknowledgement
---------------

SCIFYsim is a development carried out in the context of the SCIFY project. SCIFY
has received funding from the European Research Council (ERC) under the
European Union's Horizon 2020 research and innovation program (grant agreement No 866070).  
For more information about the SCIFY project, visit:
`this page<http://www.biosignatures.ulg.ac.be/ddefrere/scify.php>`_

Dependencies:
-------------

The following packages are required:

 - `kernuller<https://github.com/rlaugier/kernuller>`_ # https://github.com/rlaugier/kernuller
 - numpy
 - sympy
 - scipy
 - numexpr
 - matplotlib
 - tqdm
 - lmfit
 - astropy
 - astroplan
 - astroquery

Optional:
- dask
- jupyterlab

Recommandation for installation:
--------------------------------

>> python setup.py install

