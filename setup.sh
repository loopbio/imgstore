#!/usr/bin/env bash
#
# Creates a conda environment with the required dependencies:
# See: https://conda.io/docs/
#
# Running this script will create a new conda installation in the directory "miniconda".
# If you have conda already installed and do not want a redundant installation,
# make sure conda is in PATH and call this script like this:
#   I_HAVE_CONDA=YES ./setup.sh
# In this case, running this script is equivalent to the standard conda command:
#   conda env create --name imgstore --file environment.yaml
#
# Once the directory is created, activate the conda environment:
#   source activate imgstore
#

set -e

echo "Installing imgstore environment."

# Work in the recipe dir
pushd `dirname $0` > /dev/null
myDir=`pwd`
popd > /dev/null
cd ${myDir}

# Install conda if needed and make sure it is up-to-date
I_HAVE_CONDA=${I_HAVE_CONDA:-0}
if [ ${I_HAVE_CONDA} -eq 0 ]; then

  # Warn if conda is already in the system
  if hash conda 2>/dev/null; then
    condaPath=$(dirname `which conda`)
    echo "
----------------------------------------
WARNING: conda is going got be installed anew in:
   ${myDir}/miniconda
However, you seem to have it already installed in:
   ${condaPath}
If this is not what you want, please kill this script (Control-C on linux, Meta-C in mac)
Then call it like:
   I_HAVE_CONDA=YES ./setup.sh
----------------------------------------

"
    sleep 20
  fi

  # Find out what OS we are in
  if [ "$(uname)" == 'Darwin' ]; then
    OS='Mac'
  elif [ "$(expr substr $(uname -s) 1 5)" == 'Linux' ]; then
    OS='Linux'
  elif [ "$(expr substr $(uname -s) 1 10)" == 'MINGW32_NT' ]; then
    OS='Cygwin'
  else
    echo "Your platform ($(uname -a)) is not supported."
    exit 1
  fi

  # Find out the miniconda download URL
  if [ ${OS} == 'Mac' ]; then
    MINICONDA_URL=https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh
  elif [ ${OS} == 'Linux' ]; then
    MINICONDA_URL=https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
  else
    echo "Miniconda URL for your OS is not configured"
    exit 1
  fi

  # Download miniconda
  curl ${MINICONDA_URL} > miniconda.sh
  # Install miniconda under the current root
  bash miniconda.sh -b -p ${myDir}/miniconda
  rm -f miniconda.sh
  # Configure the PATH
  export PATH="${myDir}/miniconda/bin:${PATH}"
  hash -r
  # Update conda
  conda update --yes conda
  # Log
  conda info -a
fi

# Create the conda environment
conda env create -f environment.yml

# Give some hints on how to continue
hash -r
condaPath=$(dirname `which conda`)
echo "

----------------------------------------
Environment created. You can activate it by running:
  export PATH=${condaPath}:\${PATH}
  source activate imgstore
  imgstore-view --help
----------------------------------------

"
