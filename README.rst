SCIFYsim: a python package for interferometric nulling simulation
=================================================================

It is a python module designed to simulate the behaviour of high-contrast interferometric instruments, both on the ground and in space.
SCIFYsim aims at replacing GENIEsim for all the simulations in the SCIFY project.

- Absil et al. (2006), A&A, 448, 787-800.

Acknowledgement
---------------

SCIFYsim is a development carried out in the context of the SCIFY project. SCIFY
has received funding from the European Research Council (ERC) under the
European Union's Horizon 2020 research and innovation program (grant agreement No 866070).  
For more information about the SCIFY project, visit:
http://www.biosignatures.ulg.ac.be/ddefrere/scify.php

Dependencies:
-----------

The following packages are required:

 - numpy
 - sympy
 - scipy
 - matplotlib
 - tqdm
 - kernuller # https://github.com/rlaugier/kernuller
 - astropy==4.0
 - astroplan>=0.6

Recommandation for installation:
-------------------------------

>> python setup.py install
