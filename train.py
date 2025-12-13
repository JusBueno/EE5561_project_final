
import pickle
import time
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from src.data_preparation import *
from src.criterion import *
from src.testing_functions import *
from src.network_3D import REF_VAE_UNET_3D, VAE_UNET_3D_M01
from src.network_2D import VAE_UNET_2D_M01
from config import Configs, save_configs

#=========== SETUP PARAMETERS ===============

params = Configs().parse()

label = params.folder

#Directory for output results
results_path = Path('training_results')/params.folder
resume_training = (results_path/"checkpoint.pth.tar").is_file() and params.resume
results_path.mkdir(parents=True, exist_ok=True)

if resume_training: #Load existing params
    with open(results_path/'params.pkl', 'rb') as f:
        params = pickle.load(f)
    checkpoint = torch.load(results_path/"checkpoint.pth.tar", weights_only = False) 
    validation_metrics = np.load(results_path / "validation_metrics.npy")
    training_metrics = np.load(results_path / "training_metrics.npy")
    best_val_dice_loss = checkpoint['best_dice']
    best_epoch_num = checkpoint['epoch']
    params.start_epoch = checkpoint['epoch'] + 1
else:
    with open(results_path /'params.pkl', 'wb') as f:
        pickle.dump(params, f)
    save_configs(params, results_path /'params.txt')
    best_val_dice_loss = np.inf
    validation_metrics = np.zeros((params.num_epochs,3))
    training_metrics = np.zeros((params.num_epochs,3))
    best_epoch_num = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("Warning: Running on CPU")
    
data_path = '../BRATS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'

epoch_times_file = results_path / "epoch_times.npy"
if resume_training and epoch_times_file.exists():
    epoch_times = np.load(epoch_times_file)
else:
    epoch_times = np.zeros(params.num_epochs)

train_times_file = results_path / "train_times.npy"
if resume_training and train_times_file.exists():
    train_times = np.load(train_times_file)
else:
    train_times = np.zeros(params.num_epochs)

val_times_file = results_path / "val_times.npy"
if resume_training and val_times_file.exists():
    val_times = np.load(val_times_file)
else:
    val_times = np.zeros(params.num_epochs)

gpu_mem_file = results_path / "gpu_memory_usage.npy"
if resume_training and gpu_mem_file.exists():
    gpu_memory_usage = np.load(gpu_mem_file)
else:
    gpu_memory_usage = np.zeros(params.num_epochs)


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
input_shape = np.asarray(dataset.input_dim)
output_shape = dataset.output_dim


if params.net == "REF_US":
    model = REF_VAE_UNET_3D(in_channels=inChans, input_dim=np.asarray(output_shape[-3:], dtype=np.int64), num_classes=4, VAE_enable=params.VAE_enable)
elif params.net == "VAE_3D":
    model = VAE_UNET_3D_M01(in_channels=inChans, input_dim=np.asarray(output_shape[-3:], dtype=np.int64), num_classes=4, VAE_enable=params.VAE_enable, HR_layers = params.HR_layers)
elif params.net == "VAE_2D":
    model = VAE_UNET_2D_M01(in_channels=inChans*params.slab_dim, input_dim=np.asarray(output_shape[-2:], dtype=np.int64), num_classes=4, VAE_enable=True, HR_layers=params.HR_layers, fusion=params.fusion)
else:
    raise ValueError(f"Unknown network type: {params.net}")
model = model.to(device)

# Estimate network size
if isinstance(model, nn.DataParallel):
    model_to_size = model.module
else:
    model_to_size = model
param_size = sum(p.numel() * p.element_size() for p in model_to_size.parameters())
buffer_size = sum(b.numel() * b.element_size() for b in model_to_size.buffers())
model_size_bytes = param_size + buffer_size

print(f"Model size: {model_size_bytes:.4f} bytes")

with open(results_path / "model_size_bytes.txt", "w") as f:
    f.write(f"{model_size_bytes:.4f}\n")


criterion = CombinedLoss(params)

optimizer = torch.optim.Adam(model.parameters(), lr = params.LR, weight_decay = 1e-5)
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

    # Reset memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params.num_epochs} [Training]")

    train_start = time.time() # To measure train time only
    for out_imgs, inp_imgs, mask in train_bar:

        seg_pred, rec_pred, y_mid = model(inp_imgs)
        combined_loss, dice_loss, l2_loss, kl_div = criterion(seg_pred, mask, rec_pred, out_imgs, y_mid)
        training_metrics[epoch,0] += dice_loss
        training_metrics[epoch,1] += l2_loss
        training_metrics[epoch,2] += kl_div

        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()

        train_bar.set_postfix(loss=combined_loss.item())
    
    training_metrics[epoch,:] = training_metrics[epoch,:] / len(train_bar)
    np.save(results_path / "training_metrics.npy", training_metrics)

    # Save train time
    train_end = time.time()
    train_times[epoch] = train_end - train_start
    np.save(results_path / "train_times.npy", train_times)

    #---Validation
    if(params.validation):
        val_start = time.time() # To measure validation time
        
        validation_metrics[epoch,:] = test_model(model, val_loader, VAE_enable = params.VAE_enable, UNET_enable = True, logvar_out=True)
        
        np.save(results_path / "validation_metrics.npy", validation_metrics)
        
        # Save validation time
        val_end = time.time()
        val_times[epoch] = val_end - val_start
        np.save(results_path / "val_times.npy", val_times)

        dice_loss = validation_metrics[epoch,0]
        best_epoch = dice_loss < best_val_dice_loss
        if(best_epoch):
            best_val_dice_loss = dice_loss
            best_epoch_num = epoch
        plot_loss_curves(results_path, validation_metrics, training_metrics, epoch, params.VAE_enable, True, params.net)

        #Break if performance doesn't increase after (params.val_patience) epochs
        if(epoch - best_epoch_num > params.val_patience):
            break
            
    checkpoint = {
        'best_dice': best_val_dice_loss,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }

     #---Logging model checkpoint 
    if(params.save_model_each_epoch):
        torch.save(checkpoint, results_path / "checkpoint.pth.tar")
    
    if(best_epoch and params.save_model_each_epoch):
        torch.save(checkpoint, results_path / "best_checkpoint.pth.tar")
        plot_examples(model, val_dataset, range(10), results_path, VAE_enable = params.VAE_enable, UNET_enable = True, threeD = params.threeD)

    if device.type == "cuda":
        torch.cuda.synchronize()
        gpu_memory_usage[epoch] = torch.cuda.max_memory_allocated()
        np.save(results_path / "gpu_memory_usage.npy", gpu_memory_usage)

    epoch_end = time.time()
    epoch_times[epoch] = epoch_end - epoch_start
    np.save(results_path / "epoch_times.npy", epoch_times)

    epoch += 1
    scheduler.step() #Adjust learning rate
    if params.VAE_warmup:
        criterion.kl_annealer.step()
        criterion.recon_annealer.step()
    
