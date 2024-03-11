# sfm_autoencoder

## Local feature compression using autoencoders - SfM tests

Design a compression strategy for local SURF descriptors using autoencoders. Training data can be generated using the images of dataset Portello
and Castle. Testing must be done on dataset FountainP-11 and Tiso (available at https://github.com/openMVG/SfM_quality_evaluation/tree/master/
Benchmarking_Camera_Calibration_2008 and http://www.dei.unipd.it/
~sim1mil/materiale/3Drecon/). Software must be implemented in MATLAB, Keras or Pytorch.

Testing on 3D reconstruction using SfM The reconstructed descriptors (only for the test set) are used to perform a SfM reconstruction
using COLMAP (using the two test dataset).

Programming languages: MATLAB/Python/C++.

To train the autoencoder with a variable number of training datasets: 

```sh
$ main.py --train /path/to/data1 [/path/to/data2 ...]
```
No descriptors or matches need to be saved for training purposes only. The network is saved in the project root folder: ROOT/default/enc_parameters.torch and ROOT/defaultdec_parameters.torch 

To compute the surf descriptors and matches on any number of datasets:

```sh
$ main.py /path/to/data1 [/path/to/data2 ...]
```

To compute surf descriptors and compute matches on reconstructed descriptors on any number of datasets:

```sh
$ main.py --recon /path/to/data1 [/path/to/data2 ...]
```
In both cases the dummy sift descriptors are saved in the image folders of the datasets directly. The file containing the matches can be found in ROOT/default/matches_test*.txt

