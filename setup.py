import sys

from setuptools import setup

setup(name='scifysim',
      version='0.2.1', # defined in the __init__ module
      description='A python tool for simulating high contrast interferometric observations from the ground or in space.',
      url='--',
      author='Romain Laugier',
      author_email='romain.laugier@oca.eu',
      license='',
      classifiers=[
          'Development Status :: 3 - pre-Alpha',
          'Intended Audience :: Professional Astronomers',
          'Topic :: High Angular Resolution Astronomy :: Interferometry',
          'Programming Language :: Python :: 3.7'
      ],
      packages=['scifysim'],
      install_requires=[
          'numpy', 'sympy', 'scipy', 'matplotlib', 'astropy','tqdm', 'astroplan', 'kernuller', 'lmfit', 'numexpr', 'astroquery'
      ],
      include_package_data=True,
      zip_safe=False)