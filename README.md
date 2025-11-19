## EE5561 Project - Initial README

# Get started

To run create a conda environment, install requirements.txt, and install pytorch using:

`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`

# Running a training session
To run a training session, the script accepts one required argument and several optional arguments. The required argument specifies the folder where all output from the training session will be stored. This includes logs, checkpoints, configuration files, and any generated results. If the folder does not exist, it will be created automatically.

**The general command format is:**

python train.py <folder> [optional arguments]

**Required positional argument:**

folder : Name or path of the folder where the training session will be saved.

**Optional arguments:**

--resume or --start_new : Whether to resume training from existing checkpoint. Default is --resume

--net : Network type to use. Default is "REF".

--VAE_enable or --VAE_disable : Whether to use the VAE. Default it --VAE_enable 

--num_epochs : Number of epochs to train. Default is 300.

--LR : Learning rate. Default is 1e-4.

--batch : Batch size. Default is 1.

--degradation_type : Type of image degradation to apply. Default is "downsampling".

--downsamp_type : Type of downsampling interpolation. Default is "bilinear".

--ds_ratio : Downsampling ratio. Default is 1.


**Examples:**

Run training with default settings:

`python train.py experiment1`

Run training with more epochs and VAE disabled:

`python train.py experiment1 --num_epochs 500 --VAE_enable False`

Use a different network and bicubic downsampling:

`python train.py experimentA --net UNET --downsamp_type bicubic --ds_ratio 2`

Start a fresh training session even if checkpoints exist in the folder:

`python train.py output_folder --resume False`

The output folder will contain all training-related files, making each session fully reproducible and self-contained.

**More examples:**

`
python train.py MOD_02_TEST_01 --start_new --net MOD_02 --VAE_enable --ds_ratio 2
`

`
python train.py MOD_02_TEST_02 --start_new --net MOD_02 --VAE_disable --ds_ratio 2
`

For selecting gpu for execution:

`CUDA_VISIBLE_DEVICES=0`

