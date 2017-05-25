IMGStore houses your video frames
=================================

IMGStore allows to read and write video in loopbio's flexible blah

Example
-------

Put a pretty store online and write some code here...

```python
from imgstore import new_for_filename
...
store = new_for_filename('path/to/meta.yaml')
store.get_next_framenumber()
...
```

Install
-------

We recommend installing using the conda package manager. [Install
conda](https://conda.io/miniconda.html) (also available with the
[anaconda scientific python distribution](https://www.continuum.io/downloads)).
Then run:

```sh
conda install -c loopbio imgstore
```


Alternative install: pypi
-------------------------

Most of *imgstore* dependencies are in the python package index (pypi),
with the honorable exception of *opencv*. If you have a python environment
with opencv already installed on it, you should be able to install imgstore
with a command like:

```sh
pip install imgstore
```

Install a development version
-----------------------------

You can also install a development version directly from github:

```sh
git clone https://github.com/loopbio/imgstore.git
cd imgstore

# Create a conda environment...
conda env create -f environment.yml

# ... or update your existing conda environment...
conda env update -n <your-env-name> -f environment.yml

# ... or install using pip
pip install -e .
```

Once installed, you can run the test suite (including pep8 checks and
coverage) with a command like this:

```sh
py.test -v -rs --doctest-modules --pep8 --cov --cov-report term-missing imgstore --pyargs imgstore
```
