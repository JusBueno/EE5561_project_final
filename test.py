#======For testing trained network=======

import sys
from tqdm import tqdm 
from src.data_preparation import *
from src.network import *
from src.criterion import *
from src.testing_functions import *
from src.reference_net import *
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
import pickle
from config import Training_Parameters

#=========== SETUP PARAMETERS ===============

label = "ref_net_2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data_path = '../../BRATS20/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/'
data_path = '../BRATS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
results_path = Path('training_results')/label

checkpoint_path = results_path/"checkpoint.pth.tar"
checkpoint = torch.load(checkpoint_path)

with open(results_path/'params.pkl', 'rb') as f:
    params = pickle.load(f)

params.slab_dim = params.num_slices
params.slabs_per_volume = params.slices_per_volume
params.VAE_enable = True
    
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

#=========== SETUP MODEL ===============

inChans = 4; seg_outChans = 3
input_shape = (inChans,params.slab_dim, 240, 240)

#With VAE branch
if params.net == "VAE_2D":
    model = VAE_UNET(params.slab_dim, input_dim=dataset.input_dim, HR_dim=dataset.output_dim)
elif params.net == "UNET_2D":
    model = UNET(params.slab_dim)
elif params.net == "ref_3D":
    model = NvNet(inChans, input_shape, seg_outChans, "relu", "group_normalization", params.VAE_enable, mode='trilinear')


model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

print("ready to test")

plot_examples(model, val_dataset, range(10), results_path, params.net)
metrics = test_model(model, val_loader, params.net)


