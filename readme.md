# Note

I can no longer find the link to this content. I believe it was on Zenodo, but
cannot confirm that. I'm putting it on GitHub for temporary reference.

# Supplement to _Full-sky Cosmic Microwave Background Foreground Cleaning Using Machine Learning_

This work serves as a supplement to the paper titled
_Full-sky Cosmic Microwave Background Foreground Cleaning Using Machine Learning_.

This archive contains the code used to produce the training simulations,
patches for the Planck Sky Model code, a few simulation examples, the source
code for the neural network model, the trained model weights, and the notebooks
and data files used to analyze the results and produce the plots in the paper.


## Simulations

The `simulations` directory contains the script used to run the Planck Sky Model
simulations used to train the neural network model. It also contains patches
for v1.7.8 of the Planck Sky Model code such that it can use the 2018 Planck
reduced instrument model (`psm_v1-7-8_patch_rimo.diff`) and can use Model 8 of
Finkbeiner et al. (1999) for modeling thermal dust (`psm_v1-7-8_patch_dust.diff`).

The directory also contains one example from the test set for the neural
network model; a simulation is provided for the simulations run using Model 7
of Finkbeiner et al. (1999) for thermal dust as well as for simulations run
using Model 8. Also included are files containing the normalization data.


## Code

The neural network model code is in the `code` directory. The `scnn`
subdirectory contains the code used to implement the DeepSphere architecture
and Concrete dropout, while the model itself is implemented in the `model.py`
script. The `process-sims512.py` script adds noise to the simulations,
normalizes them, and reorders them. The `train-component-separation512.py`
script runs the model training procedure, and the
`eval-component-separation512.py` and `eval-component-separation512_dust.py`
scripts evaluate the model. The shell scripts are helper scripts to run the
training and evaluation, which primarily exist to avoid issues due to the large
protocol buffers used. The neural network model was built using TensorFlow v2.1
and the TensorFlow Large Model Support library (patch for TensorFlow v2.1).


## Model

The `model` directory contains the two trained models, one trained with
simulations using Model 7 of Finkbeiner et al. (1999) for thermal dust and the
other using Model 8. Also included are the log files from the training process.
Each model subdirectory contains the evaluation results for Planck frequency
maps and one example simulation from the test set. The subdirectory
corresponding to the model trained using dust Model 7 also includes the results
of evaluating a simulation that makes use of dust Model 8. The `*_raw.npz`
files contain the results of the 100 separate evaluations, while the `*.npz`
files that do not include `_raw` contain the averaged maps.


## Notebooks

The `notebooks` directory contain the Jupyter notebooks used for evaluating the
results and producing the plots used in the paper. The `calc-stats-spectra.ipynb`
notebook cannot be run with the data provided in this archive, since it
requires the evaluation results of 100 maps from the training set to evaluate
the angular power spectrum bias; these data were excluded since they are ~200GB.
The summary data produced by said notebook are included, so the plotting
notebooks can be executed (after adjusting data locations for items such as
Planck maps).


## License

This work is placed into the public domain via the
[CC0 1.0 public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/)
except for the contents of the `code/scnn` directory, which are made available
under the [MIT License](https://opensource.org/licenses/MIT) and are based on
code by Michaël Defferrard, Nathanaël Perraudin, Austin Clyde, and Yarin Gal
(see headers of these files for more details).


## Credits

This work was produced by Matthew Petroff
([ORCID:0000-0002-4436-4215](https://orcid.org/0000-0002-4436-4215)). Parts of
the code are based on work by Michaël Defferrard, Nathanaël Perraudin,
Austin Clyde, and Yarin Gal.
