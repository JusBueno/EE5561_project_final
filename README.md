## EE5561 Project - README
### Group members: Rafael Avelar, Justin Bueno, Antonio Jimenez
Our project trained a neural network to perform segmentation on the BRATS2020 dataset. The project's aim was to develop a network architecture that could efficiently segment downsampled inputs. Our baseline architecture used a UNET combined with a VAE decoder, as a form of regularization. We experimented with many variations of this, and conducted a through ablation study and hyperparameter search to find out the impact of the VAE. 

# Get started
This project needs Python 3.10+. To run create a conda environment, install dependencies from requirements.txt

# Running inference and validation
To run examples using pre-trained models and get their validation accuracy, see demo.ipynb. You do not need to download the full dataset for this notebook.

# Dataset download
To download the full dataset, follow the instructions and execute dataset_download.ipynb.

# Running a training session
To train a model from scratch,
`train.py` accepts one required argument and several optional arguments. The required argument specifies the folder where all output from the training session will be stored. This includes logs, checkpoints, configuration files, and any generated results. If the folder does not exist, it will be created automatically. You need to download the full dataset to use this script.

**The general command format is:**

python train.py <folder> [optional arguments]

**Required positional argument:**

folder : Name or path of the folder where the training session will be saved.

**Optional arguments:**

**System Controls**

--resume or --start_new : Whether to resume training from existing checkpoint. Default is --resume

--net : Network type to use. Default is "REF_US". Options are [REF_US, VAE_3D, VAE_2D] REF_US is the reference network, VAE_3D and VAE_2D are the networks that
can take downsampled inputs.

--num_epochs : Number of epochs to train. Default is 300.

--LR : Learning rate. Default is 1e-4.

--batch : Batch size. Default is 1.

**Cropping**
--crop or --no_crop : Whether to crop the images. Default is True

**VAE-Related**
--VAE_enable or --VAE_disable : Whether to use the VAE. Default it --VAE_enable 

--VAE_warmup : Activates VAE weight scheduling. Default is deactivated

--k1 : MSE Reconstruction weight. Default is 0.1
--k2 : KL divergence weight. Default is 0.1

**Downsizing techniques**

--downsamp_type : Type of downsampling interpolation. Default is "bilinear". Options are [bilinear, bicubic, wavelet, ds_filter, ds]

--ds_ratio : Downsampling ratio. Default is 1. Options are [1,2,4]


**2.5D parameters**

--slab_dim : Number of slices. Default is 144. Must specify for 2D, for which options are [3,5,7,9]. Should be fixed to default when 3D

--fusion : 2D Fusion type. Default is None. Options are [None, Slab, Modality, Hybrid]





**Examples:**

Run training with default settings:

`python train.py experiment1`

Run training with more epochs and VAE disabled:

`python train.py experiment1 --num_epochs 500 --VAE_enable False`

Use the modified network using downsampled data and bicubic downsampling (x2):

`python train.py experimentA --net VAE_3D --downsamp_type bicubic --ds_ratio 2`

Start a fresh training session even if checkpoints exist in the folder:

`python train.py output_folder --start_new`

The output folder will contain all training-related files, making each session fully reproducible and self-contained.

**More examples:**

New training session using REF_US network with VAE, saved in a folder named REF_US_TEST_01

`
python train.py REF_US_TEST_01 --start_new --net REF_US --VAE_enable 
`

Resume training session using VAE_3D network with VAE, learning rate of 1e-5, reconstruction weight of 0.01, KL weight of 0.001, saved in a folder named VAE_3D_TEST_01

`
python train.py VAE_3D_TEST_01 --resume --net VAE_3D --VAE_enable --LR 0.00001 --k1 0.01 --k2 0.001 
`

Resume training session using VAE_2D network without VAE, downsampling with wavelet x4, with slab fusion, 5 slices saved in a folder named VAE_2D_TEST_63

`
python train.py VAE_3D_TEST_01 --resume --net VAE_2D --VAE_disabled --downsamp_type wavelet --ds_ratio 4 --fusion Slab --slab_dim 5 
`

New training session using VAE_3D network with VAE, with weight scheduling, and without cropping saved in a folder named VAE_3D_TEST_30

`
python train.py VAE_3D_TEST_30 --start_new --net VAE_3D --VAE_enable --VAE_warmup --no_crop 
`

