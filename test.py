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
from config import Training_Parameters, parse_args


#=========== SETUP PARAMETERS ===============

args = parse_args()
label = args.folder

#Directory for output results
results_path = Path('training_results')/args.folder
if not (results_path/"params.pkl").is_file():
    print("ERROR: File not found")
    sys.exit(1)

with open(results_path/'params.pkl', 'rb') as f:
    params = pickle.load(f)

validation_metrics = np.load(results_path / "validation_metrics.npy")
training_metrics = np.load(results_path / "training_metrics.npy")
best_val_dice = validation_metrics.max(axis=0)[0]

epoch = 100

if params.VAE_enable:
    fig,  ax = plt.subplots(1,3,figsize = (8,3.5))
    title_list = ["Dice coefficient", "MSE", "KL Divergence"]
    
    ax[0].plot(1-(training_metrics[:epoch+1, 0]/3), label='Training')
    ax[0].plot(validation_metrics[:epoch+1, 0], label='Validation')
    ax[0].set_xlabel("Epochs", fontsize = 13)
    ax[0].set_ylabel("Dice coefficient", fontsize = 13)
    #ax[0].legend(fontsize = 10)
    
    for i in range(1,3):
        ax[i].plot(validation_metrics[:epoch+1, i], label='Validation')
        ax[i].plot(training_metrics[:epoch+1, i], label='Training')
        ax[i].set_xlabel("Epochs", fontsize = 13)
        ax[i].set_ylabel(title_list[i], fontsize = 13)
    
    ax[2].legend(fontsize = 10)
    fig.suptitle(f"Network: {params.net}, VAE is {params.VAE_enable}", fontsize=13)
   
    
else:
    fig,  ax = plt.subplots(1,1,figsize = (3.5,3.5))
    
    ax.plot(1-training_metrics[:epoch+1, 0]/3, label='Training')
    ax.plot(validation_metrics[:epoch+1, 0], label='Validation')
    ax.set_xlabel("Epochs", fontsize = 13)
    ax.set_ylabel("Dice coefficient", fontsize = 13)
    ax.legend(fontsize = 10)
    ax.set_title(f"Network: {params.net}, VAE is {params.VAE_enable}", fontsize=13)


fig.tight_layout()
fig.savefig(results_path/"loss_curves.png")


# Plot time metrics:
# Load the data
results_path = Path("training_results") / sys.argv[1]
save_path = results_path / "epoch_times_plot.png"
epoch_times_path = results_path / "epoch_times.npy"
epoch_times = np.load(epoch_times_path)

# Remove zeros
last_nonzero_idx = np.nonzero(epoch_times)[0][-1]
epoch_times_to_plot = epoch_times[:last_nonzero_idx + 1]
epochs = np.arange(1, len(epoch_times_to_plot) + 1)

# Some metrics
avg = np.mean(epoch_times_to_plot)
max = np.max(epoch_times_to_plot)
min = np.min(epoch_times_to_plot)
std = np.std(epoch_times_to_plot,ddof=1)/np.sqrt(len(epoch_times_to_plot))
stats = f"Avg:{avg:.1f} secs\nsigma:{std:.2f} secs\nMax:{max:.1f} secs\nMin:{min:.1f} secs"

plt.figure()
plt.plot(epochs, epoch_times_to_plot)
plt.xlabel("Epoch")
plt.ylabel("Time per epoch (seconds)")
plt.title(f"Training time per epoch: {sys.argv[1]}")
plt.annotate(stats, (0.01,0.85),xycoords='axes fraction')
plt.tight_layout()
plt.savefig(save_path)
#plt.show()





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


    
