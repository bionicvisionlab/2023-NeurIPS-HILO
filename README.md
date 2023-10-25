# 2023-NeurIPS-HILO
Making a repository for this so that as things change (p2p integration etc), code will still be available and work properly. Also because noone ever seems to be able to find NeurIPS supplemental. Will change to public once it's ready.

## Instructions for Installation
Our code was developed using Python 3.10; it might work with previous versions, but if you run into any errors, try Python 3.10.

We include a `requirements.txt` file, which can be run to obtain all of the python dependencies. We also included `python_env.txt` for reference, with a complete list of all python packages installed.

However, we also use the following software, which requires additional instalation steps:
- [pulse2percept](https://pulse2percept.readthedocs.io/). This is an open source package for simulating visual prostheses. It can be installed with pip (and is in requirements.txt) and likely will install without issue. But it also depends on some non-python packages (e.g. 
gcc). Many systems have these installed already. But if your installation fails, please go through the installation instructions for pulse2percept at https://pulse2percept.readthedocs.io/en/stable/install.html
- Tensorflow. Tensorflow can also be installed with pip, but you may have to set up CUDA if you want it to use a GPU. See https://www.tensorflow.org/install/pip
- Matlab. All Bayesian optimization is performed in Matlab, using existing open source Bayesian optimization and Gaussian Process toolboxes (https://github.com/TristanFauvel/BO_toolbox, https://github.com/TristanFauvel/GP_toolbox). If you do not have, or do not wish to install Matlab, then you can stil run all code for the deep stimulus encoder and phosphene model, and see the results from HILO in `HILO.ipynb` from us running the notebook. The following section provides more details on installing Matlab.

### Matlab installation
Unfortunately, matlab is not freely available. However, most universities have licensing deals to get it for free. We used matlab version R2022a, but other versions might also work. You also will need the deep learning toolbox, computer vision toolbox, and machine learning and statistics toolbox (provided with matlab as additional downloads)

Once installed, there are two additional steps that must be performed to set up matlab with our code: 
1) Install the python matlab engine. This involves running `pip install setup.py` on a setup.py file provided with your matlab installation. See https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html.
2) Download the external, open source Bayesian optimization and Gaussian process toolboxes to the `code/matlab` folder. \
`cd code/matlab` \
`git clone git@github.com:TristanFauvel/BO_toolbox.git` \
`git clone git@github.com:TristanFauvel/GP_toolbox.git`


A this point, you should be ready to run any of our code or the HILO demo notebook. 
