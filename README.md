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

**System Controls**

--resume or --start_new : Whether to resume training from existing checkpoint. Default is --resume

--net : Network type to use. Default is "REF_US". Options are [REF_US, VAE_3D, VAE_2D]

--num_epochs : Number of epochs to train. Default is 300.

--LR : Learning rate. Default is 1e-4.

--batch : Batch size. Default is 1.

**Cropping**
--crop or --no_crop : Whether to crop the images. Default is True

**VAE-Related**
--VAE_enable or --VAE_disable : Whether to use the VAE. Default it --VAE_enable 

--VAE_warmup : Activates VAE weight scheduling. Default is deactivated

--k1 : Reconstruction weight. Default is 0.1
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

For selecting gpu for execution:

`CUDA_VISIBLE_DEVICES=0`

# Test and Implementation

1. Determine the best performing LR, stop with plateau
(1) REF_US, LR: 1e-4, with VAE *    | Dice = 0.744
(2) REF_US, LR: 1e-4, without VAE * | Dice = 0.765
(3) REF_US, LR: 5e-5, with VAE      | Dice = 0.799
(4) REF_US, LR: 5e-5, without VAE   | Dice = 0.807
(5) REF_US, LR: 1e-5, with VAE *    | Dice = 0.790
(6) REF_US, LR: 1e-5, without VAE * | Dice = 0.796

2. Determine the best performing weight schema (use best performing LR)
(7) REF_US, [1, 0.1, 0.1], with VAE 
(8) REF_US, [1, 0.1, 0.01], with VAE *    | Dice = 0.800
(9) REF_US, [1, 0.1, 0.001], with VAE *   | Dice = 0.795
(10) REF_US, [1, 0.05, 0.01], with VAE    | Dice = 0.791
(11) REF_US, [1, 0.05, 0.001], with VAE * | Dice = 0.793
(12) REF_US, [1, 0.05, 0.0001], with VAE  | Dice = 0.799
(13) REF_US, [1, 0.01, 0.001], with VAE   | Dice = 0.810
(14) REF_US, [1, 0.01, 0.0001], with VAE  | Dice = 0.791

3. VAE 3D with downsized images - Bilinear x2 - Learning Rate
(15) VAE_3D, LR: 1e-4, with VAE *
(16) VAE_3D, LR: 1e-4, without VAE *
(17) VAE_3D, LR: 5e-5, with VAE 
(18) VAE_3D, LR: 5e-5, without VAE
(19) VAE_3D, LR: 1e-5, with VAE
(20) VAE_3D, LR: 1e-5, without VAE

4. VAE 3D with downsized images - Bilinear x2 - Weight Schema
(21) VAE_3D, [1, 0.1, 0.1], with VAE *
(22) VAE_3D, [1, 0.1, 0.01], with VAE *   | Dice = 0.784
(23) VAE_3D, [1, 0.1, 0.001], with VAE *  | Dice = 0.767
(24) VAE_3D, [1, 0.05, 0.01], with VAE    | Dice = 0.770
(25) VAE_3D, [1, 0.05, 0.001], with VAE * | Dice = 0.782
(26) VAE_3D, [1, 0.05, 0.0001], with VAE
(27) VAE_3D, [1, 0.01, 0.001], with VAE
(28) VAE_3D, [1, 0.01, 0.0001], with VAE

5. VAE 3D with downsized images - Bilinear x2 - Weight Scheduling *
(29) RECON_W - Linear  0.01 to 0.05 (5 epochs), then constant 0.05
     KL_W - constant 0 (2 epochs), then Linear 0 - 0.001 (15 epochs), then constant 0.001
Note: Values can be based on previous results

6. VAE 3D with downsized images (using best LR and Weights, or NO VAE if proved unnecessary to this point) - DS Methods
(30) Bilinear x2 (Done by this point) * 
(31) Bilinear x4 * 
(32) Wavelet x2 *
(34) Wavelet x4 *
(35) DS with Filter x2
(36) DS with Filter x4
(37) Bicubic x2
(38) Bicubic x4
(39) DS x2
(40) DS x4

7. VAE 2D - Bilinear x2, 5 Slices, None Fusion - LR Exploration
(41) VAE_2D, LR: 1e-4, with VAE *
(42) VAE_2D, LR: 1e-4, without VAE *
(43) VAE_2D, LR: 5e-5, with VAE
(44) VAE_2D, LR: 5e-5, without VAE
(45) VAE_2D, LR: 1e-5, with VAE
(46) VAE_2D, LR: 1e-5, without VAE

8. VAE 2D with downsized images - Bilinear x2, 5 Slices, None Fusion - Weight Schema
(47) VAE_2D, [1, 0.1, 0.1], with VAE *
(48) VAE_2D, [1, 0.1, 0.01], with VAE *
(49) VAE_2D, [1, 0.1, 0.001], with VAE *
(50) VAE_2D, [1, 0.05, 0.01], with VAE
(51) VAE_2D, [1, 0.05, 0.001], with VAE *
(52) VAE_2D, [1, 0.05, 0.0001], with VAE
(53) VAE_2D, [1, 0.01, 0.001], with VAE
(54) VAE_2D, [1, 0.01, 0.0001], with VAE

9. VAE 2D with downsized images - Bilinear x2, 5 Slices, None Fusion - Weight Scheduling *
(55) RECON_W - Linear  0.01 to 0.05 (5 epochs), then constant 0.5
     KL_W - constant 0 (2 epochs), then Linear 0 - 0.001 (15 epochs), then constant 0.001
Note: Values can be based on previous results

10. VAE 2D with downsized images - Bilinear x2, 5 Slices - Fusion Schema
(56) None (done by this point) *
(57) Slab  *
(58) Modality *
(59) Hybrid

11. VAE 2D with downsized images - Bilinear x2 - Slice Number
(60) 3 *
(61) 5 (Done by this point) *
(62) 7 *
(63) 9

12. VAE 2D with downsized images - DS Methods
(64) Bilinear x2 (Done by this point) *
(65) Bilinear x4 *
(66) Wavelet x2 *
(67) Wavelet x4 *
(68) DS with Filter x2
(69) DS with Filter x4
(70) Bicubic x2
(71) Bicubic x4
(72) DS x2
(73) DS x4
