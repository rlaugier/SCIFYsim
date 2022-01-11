Documentation recipe
--------------------
This documentation was generated through the following sequence.


.. code-block::

	cd SCIFYsim
	mkdir docs
	cd docs
	sphinx-quickstart --ext-autodoc
	subl source/conf.py
		* uncomment the reference to the repo
		* repo should be "../../scifysim/"
		* html_theme = 'classic'
		* from scifysim import version
		* release = version

	sphinx-apidoc -o source/ ../scifysim

	make clean
	make html
	touch build/html/.nojekyll

We use "classic" theme because the RTD theme fails in a number of ways (fails to display itmeized lists, and some of the text...)

