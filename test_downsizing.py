import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from src.downsample import filter_downsample
from src.img_wavelet import img_wavelet
from src.degradate_reconstruct import transform_downsample_reconstruct



import sys
import pickle
import time
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from src.data_preparation import *
from src.network import *
from src.criterion import *
from src.testing_functions import *
from src.reference_net import *
from src.reference_net_mod01 import NvNet_MOD01
from src.reference_net_mod02 import NvNet_MOD02
from src.reference_net_mod03 import NvNet_MOD03
from config import Training_Parameters, parse_args

#=========== SETUP PARAMETERS ===============

args = parse_args()
label = args.folder

#Directory for output results
results_path = Path('training_results')/args.folder
resume_training = (results_path/"checkpoint.pth.tar").is_file() and args.resume
results_path.mkdir(parents=True, exist_ok=True)


if resume_training: #Load existing params
    with open(results_path/'params.pkl', 'rb') as f:
        params = pickle.load(f)
    checkpoint = torch.load(results_path/"checkpoint.pth.tar", weights_only = False) 
    validation_metrics = np.load(results_path / "validation_metrics.npy")
    training_metrics = np.load(results_path / "training_metrics.npy")
    best_val_dice = validation_metrics.max(axis=0)[0]
else:
    #Initialize training parameters
    params = Training_Parameters(
        net=args.net,
        VAE_enable=args.VAE_enable,
        num_epochs=args.num_epochs,
        LR=args.LR,
        batch=args.batch,
        degradation_type=args.degradation_type,
        downsamp_type=args.downsamp_type,
        ds_ratio=args.ds_ratio
    )
    params.batch_size *= torch.cuda.device_count() # If parallel training
    #Save the parameters to keep track of what we ran
    with open(results_path /'params.pkl', 'wb') as f:
        pickle.dump(params, f)
    best_val_dice = 0
    validation_metrics = np.zeros((params.num_epochs,3))
    training_metrics = np.zeros((params.num_epochs,3))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("Warning: Running on CPU")
    
data_path = '../BRATS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'


epoch_times_file = results_path / "epoch_times.npy"
if resume_training and epoch_times_file.exists():
    epoch_times = np.load(epoch_times_file)
else:
    epoch_times = np.zeros(params.num_epochs)


#=========== SETUP DATASETS AND DATA LOADERS ===============

dataset = BRATS_dataset(data_path, device, params)

#Create training and validation datasets
train_size = int(params.train_ratio * len(dataset))
val_size = len(dataset) - train_size

# For reproducibility, set a random seed
g = torch.Generator().manual_seed(42) 
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=g
)

train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)


#=========== SETUP MODEL AND OPTIMIZER ===============

inChans = 4; seg_outChans = 3
input_shape = (inChans, params.slab_dim//params.ds_ratio, 240//params.ds_ratio, 240//params.ds_ratio)
output_shape = (inChans, params.slab_dim, 240, 240) # Reference input (not downsized or degraded)

#With VAE branch
if params.net == "VAE_2D":
    model = VAE_UNET(params.slab_dim, input_dim=dataset.input_dim, HR_dim=dataset.output_dim)
elif params.net == "UNET_2D":
    model = UNET(params.slab_dim)
elif params.net == "REF":
    model = NvNet(inChans, input_shape, seg_outChans, "relu", "group_normalization", params.VAE_enable, mode='trilinear')
elif params.net == "MOD_01":
    model = NvNet_MOD01(inChans, output_shape, seg_outChans, "relu", "group_normalization", params.VAE_enable, mode='trilinear', HR_layers = params.HR_layers)
elif params.net == "MOD_02":
    model = NvNet_MOD02(inChans, output_shape, seg_outChans, "relu", "group_normalization", params.VAE_enable, mode='trilinear', HR_layers = params.HR_layers)
elif params.net == "MOD_03":
    model = NvNet_MOD03(inChans, output_shape, seg_outChans, "relu", "group_normalization", params.VAE_enable, mode='trilinear', HR_layers = params.HR_layers)

model = model.to(device)

if torch.cuda.device_count() >= 2:
    model = nn.DataParallel(model)  
    print(f"Parallel training with {torch.cuda.device_count()} GPUs")

criterion = CombinedLoss(VAE_enable = params.VAE_enable, separate = True)

optimizer = torch.optim.Adam(model.parameters(), lr = params.learning_rate, weight_decay = 1e-5)
lr_lambda = lambda epoch: (1 - epoch / params.num_epochs) ** 0.9
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)



if resume_training: #Load existing params
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']+1
    print(f"RESUMING TRAINING {label}, net type is {params.net}, VAE enabled is {params.VAE_enable}" )
else:
    epoch = 0
    print(f"STARTING TRAINING {label}, net type is {params.net}, VAE enabled is {params.VAE_enable}" )


#=========== TRAINING LOOP ===============


best_epoch = True

while epoch < params.num_epochs:

    epoch_start = time.time()

    model.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params.num_epochs} [Training]")

    for out_imgs, inp_imgs, mask in train_bar:

        optimizer.zero_grad()
        if params.net == "VAE_2D":
            central_index = params.slab_dim//2
            central_slice = out_imgs[:,central_index,:,:].unsqueeze(1) #Get central slice for VAE output
            seg_out, vae_out, mu, logvar = model(inp_imgs)
            combined_loss = criterion(seg_out, mask, vae_out, central_slice, mu, logvar)
        elif params.net == "UNET_2D":
            seg_out = model(inp_imgs)
            combined_loss = criterion(seg_out, mask)
        elif params.net in ["REF", "MOD_01", "MOD_02", "MOD_03"]:
            seg_pred, rec_pred, y_mid = model(inp_imgs)
            combined_loss, dice_loss, l2_loss, kl_div = criterion(seg_pred, mask, rec_pred, out_imgs, y_mid)
            training_metrics[epoch,0] += dice_loss
            training_metrics[epoch,1] += l2_loss
            training_metrics[epoch,2] += kl_div
        break
    


phantom = shepp_logan_phantom()
phantom = phantom.astype(np.float32)
phantom = resize(phantom, (256, 256), anti_aliasing=True).astype(np.float32)
phantom = central_slice

img_filtered_down = filter_downsample(phantom, 2)
img_wav = img_wavelet(phantom)
img_tv_rec = transform_downsample_reconstruct(phantom, 0.3)
img_tv_rec = np.real(img_tv_rec)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img_filtered_down, cmap="gray")
axes[0].set_title("Decimation by 2")
axes[0].axis("off")

axes[1].imshow(img_wav, cmap="gray")
axes[1].set_title("First wavelet level")
axes[1].axis("off")

axes[2].imshow(img_tv_rec, cmap="gray")
axes[2].set_title("Reconstructed from sampled Fourier space")
axes[2].axis("off")

plt.tight_layout()
plt.show()
plt.savefig('./training_results')

