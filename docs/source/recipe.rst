.. _docrecipe:

Documentation recipe
--------------------
This documentation was generated through the following sequence.


.. code-block::

	cd SCIFYsim
	mkdir docs
	cd docs
	sphinx-quickstart --ext-autodoc

	pip install renku-sphinx-theme
	subl source/conf.py
		* uncomment the reference to the repo
		* repo should be "../../scifysim/"
		* html_theme = 'renku'
		* from scifysim import version
		* release = version


	export SPHINX_APIDOC_OPTIONS=members,undoc-members,show-inheritance,special-members __init__
	sphinx-apidoc -f --separate -o source/ ../scifysim

	# We must create a log file in source so that conf.py can import SCIFYsim to get the version
	mkdir source/log

	make html
	touch build/html/.nojekyll

To compile the documentation
.. code-block::

	make clean
	make html

We use "classic" theme because the RTD theme fails in a number of ways (fails to display itmeized lists, and some of the text...)

