import sys
import pickle
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
from config import Training_Parameters


#=========== SETUP PARAMETERS ===============

#Get command line arguments
label = sys.argv[1]

#Directory for output results
results_path = Path('training_results')/label
resume_training = results_path.is_dir()
results_path.mkdir(parents=True, exist_ok=True)


with open(results_path/'params.pkl', 'rb') as f:
    params = pickle.load(f)
checkpoint = torch.load(results_path/"checkpoint.pth.tar", weights_only = False) 
validation_metrics = np.load(results_path / "training_metrics.npy")
epoch = 150

fig,  ax = plt.subplots(1,3,figsize = (8,3))
title_list = ["Dice coefficient", "MSE", "KL Divergence"]
for i in range(3):
    ax[i].plot(validation_metrics[2:epoch+1, i])
    ax[i].set_xlabel("Epochs")
    ax[i].set_ylabel(title_list[i])

fig.tight_layout()
fig.savefig(results_path/"loss_curves.png")

"""

#=========== SETUP DATASETS AND DATA LOADERS ===============

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = '../BRATS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
validation_metrics = np.zeros((params.num_epochs,3))


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

#With VAE branch
if params.net == "VAE_2D":
    model = VAE_UNET(params.slab_dim, input_dim=dataset.input_dim, HR_dim=dataset.output_dim)
elif params.net == "UNET_2D":
    model = UNET(params.slab_dim)
elif params.net == "REF":
    model = NvNet(inChans, input_shape, seg_outChans, "relu", "group_normalization", params.VAE_enable, mode='trilinear')
elif params.net == "MOD_01":
    model = NvNet_MOD01(inChans, input_shape, seg_outChans, "relu", "group_normalization", params.VAE_enable, mode='trilinear', HR_layers = params.HR_layers)
elif params.net == "MOD_02":
    model = NvNet_MOD02(inChans, input_shape, seg_outChans, "relu", "group_normalization", params.VAE_enable, mode='trilinear', HR_layers = params.HR_layers)
elif params.net == "MOD_03":
    model = NvNet_MOD03(inChans, input_shape, seg_outChans, "relu", "group_normalization", params.VAE_enable, mode='trilinear', HR_layers = params.HR_layers)

    
model = model.to(device)
criterion = CombinedLoss(VAE_enable = params.VAE_enable)

model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']

# validation_metric = test_model(model, val_loader, params.net, VAE_enable = params.VAE_enable)
# plot_examples(model, val_dataset, range(10), results_path, params.net, VAE_enable = params.VAE_enable)



"""


    
