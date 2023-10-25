[![license](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/bionicvisionlab/2023-NeurIPS-HILO/blob/master/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3458709.3458982-orange)](https://doi.org/10.48550/arXiv.2306.13104)
[![Data](https://img.shields.io/badge/data-osf.io-lightgrey.svg)](https://osf.io/pc73x/)

# Human-in-the-Loop Optimization for Deep Stimulus Encoding in Visual Prostheses
This repository houses code accompanying the paper:

> Granley, J., Fauvel, T., Chalk, M., & Beyeler, M. (2023). Human-in-the-Loop Optimization for Deep Stimulus Encoding in Visual Prostheses. _Proceedings of the Thirty-seventh Conference on Neural Information Processing Systems_ ([link](https://arxiv.org/pdf/2306.13104.pdf))


## Instructions for Installation
This code was developed using Python 3.10; it might work with previous versions, but if you run into any errors, try Python 3.10.

Included is a `requirements.txt` file, which can be run to obtain all of the python dependencies. Also included is a `python_env.txt` for reference, with a complete list of all python packages installed.

This package also uses the following software, which require additional instalation steps:
- [pulse2percept](https://pulse2percept.readthedocs.io/). This is an open source package for simulating visual prostheses. It can be installed with pip (and is in requirements.txt) and likely will install without issue. But it also depends on some non-python packages (e.g. 
gcc). Many systems have these installed already. But if your installation fails, please go through the installation instructions for pulse2percept at https://pulse2percept.readthedocs.io/en/stable/install.html
- Tensorflow. Tensorflow can also be installed with pip, but you may have to set up CUDA if you want it to use a GPU. See https://www.tensorflow.org/install/pip
- Matlab. All Bayesian optimization is performed in Matlab, using existing open source Bayesian optimization and Gaussian Process toolboxes (https://github.com/TristanFauvel/BO_toolbox, https://github.com/TristanFauvel/GP_toolbox). If you do not have, or do not wish to install Matlab, then you can stil run all code for the deep stimulus encoder and phosphene model, and see the results from HILO in `HILO.ipynb` from us running the notebook. The following section provides more details on installing Matlab.

### Matlab installation
Matlab is not freely available, however, most universities have licensing deals to get it for free. We used matlab version R2022a, but other versions might also work. 
Required matlab toolboxes (provided with matlab as additional downloads):
- deep learning toolbox,
- computer vision toolbox
- machine learning and statistics toolbox 

Once installed, there are two additional steps that must be performed to set up matlab python extension and install open source toolboxes: 
1) Install the python matlab engine. This involves running `pip install setup.py` on a setup.py file provided with your matlab installation. See https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html.
2) Download the external, open source Bayesian optimization and Gaussian process toolboxes to the `code/matlab` folder. \
`cd code/matlab` \
`git clone git@github.com:TristanFauvel/BO_toolbox.git` \
`git clone git@github.com:TristanFauvel/GP_toolbox.git`

