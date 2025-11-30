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
    fig.suptitle(f"Network: {params.net}, VAE is {params.VAE_enable}, best VAL DICE = {best_val_dice:.3f}", fontsize=13)
   
    
else:
    fig,  ax = plt.subplots(1,1,figsize = (3.5,3.5))
    
    ax.plot(1-training_metrics[:epoch+1, 0]/3, label='Training')
    ax.plot(validation_metrics[:epoch+1, 0], label='Validation')
    ax.set_xlabel("Epochs", fontsize = 13)
    ax.set_ylabel("Dice coefficient", fontsize = 13)
    ax.legend(fontsize = 10)
    ax.set_title(f"Network: {params.net}, VAE is {params.VAE_enable}, best VAL DICE = {best_val_dice:.3f}", fontsize=13)


fig.tight_layout()
fig.savefig(results_path/"loss_curves.png")


# Plot time and GPU metrics:
# Load the data
results_path = Path("training_results") / sys.argv[1]

epoch_times_path = results_path / "epoch_times.npy"
train_times_path = results_path / "train_times.npy"
val_times_path = results_path / "val_times.npy"
gpu_mem_path = results_path / "gpu_memory_usage.npy"
model_size_txt = results_path / "model_size_bytes.txt"

epoch_times = np.load(epoch_times_path)
train_times = np.load(train_times_path)
val_times = np.load(val_times_path)
gpu_mem_usage = np.load(gpu_mem_path)
with open(model_size_txt, "r") as f:
    model_size = float(f.readline().strip())

# Remove zeros
last_nonzero_idx = np.nonzero(epoch_times)[0][-1]
epoch_times = epoch_times[:last_nonzero_idx + 1]
train_times = train_times[:last_nonzero_idx + 1]
val_times = val_times[:last_nonzero_idx + 1]
gpu_mem_usage = gpu_mem_usage[:last_nonzero_idx + 1]

# Define epochs
epochs = np.arange(1, len(epoch_times) + 1)

# Time plots
save_path_times = results_path / "epoch_times_plot.png"

avg_total = np.mean(epoch_times)
std_total = np.std(epoch_times,ddof=1)/np.sqrt(len(epoch_times))
stats_total = f"Avg_total:{avg_total:.1f} secs; sigma_total:{std_total:.2f} secs\n"

avg_train = np.mean(train_times)
std_train = np.std(train_times,ddof=1)/np.sqrt(len(train_times))
stats_train = f"Avg_train:{avg_train:.1f} secs; sigma_train:{std_train:.2f} secs\n"

avg_val = np.mean(val_times)
std_val = np.std(val_times,ddof=1)/np.sqrt(len(val_times))
stats_val = f"Avg_val:{avg_val:.1f} secs; sigma_val:{std_val:.2f} secs"

stats = stats_train + stats_val + stats_total

plt.figure()
plt.plot(epochs, epoch_times, label="Total time")
plt.plot(epochs, train_times, "--", label="Train time")
plt.plot(epochs, val_times, "--", label="Validation time")
plt.xlabel("Epoch")
plt.ylabel("Time per epoch (seconds)")
plt.title(f"Train, Validation & Total time per epoch: {sys.argv[1]}")
plt.annotate(stats, (0.01,0.85),xycoords='axes fraction')
plt.legend()
plt.tight_layout()
plt.savefig(save_path_times)
#plt.show()

# GPU plot
save_path_gpu = results_path / "gpu_mem_usage_plot.png"

# converting bytes into GB
gpu_mem_usage_gb = gpu_mem_usage / 1e9
model_size_gb = model_size / 1e9

# Stats 
avg_gpu = np.mean(gpu_mem_usage_gb)
std_gpu = np.std(gpu_mem_usage_gb,ddof=1)/np.sqrt(len(gpu_mem_usage_gb))
stats_gpu = f"Avg_total:{avg_gpu:.1f} GB; sigma_total:{std_gpu:.2f} GB\n"
model_size = f"\nModel size: {model_size_gb:.1f} GB"

plt.figure()
plt.plot(epochs, gpu_mem_usage_gb)
plt.xlabel("Epoch")
plt.ylabel("Peak GPU memory (GB)")
plt.title(f"GPU peak memory usage per epoch: {sys.argv[1]}")
plt.annotate(stats_gpu+model_size, (0.01, 0.5), xycoords="axes fraction")
plt.tight_layout()
plt.savefig(save_path_gpu)
plt.close()

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


    