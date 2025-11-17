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

params = Training_Parameters() #Initialize training parameters, you can change them in src/data_preparation.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = '../BRATS20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'

#Directory for output results
results_path = Path('training_results')/label
results_path.mkdir(parents=True, exist_ok=True)

#Save the parameters to keep track of what we ran
with open(results_path /'params.pkl', 'wb') as f:
    pickle.dump(params, f)

validation_metrics = np.zeros((params.num_epochs,3))

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
    

criterion = CombinedLoss(VAE_enable = params.VAE_enable)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = params.learning_rate, weight_decay = 1e-5)

# LR Scheduler
lr_lambda = lambda epoch: (1 - epoch / params.num_epochs) ** 0.9
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

#=========== TRAINING LOOP ===============
print(f"Starting training {label}, net type is {params.net}, VAE enabled is {params.VAE_enable}" )

best_val_dice = 0
best_epoch = True

for epoch in range(params.num_epochs):

    model.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params.num_epochs} [Training]")

    for out_imgs, inp_imgs, mask in train_bar:

        optimizer.zero_grad()
        if params.net == "VAE_2D":
            central_index = params.slab_dim//2
            central_slice = out_imgs[:,central_index,:,:].unsqueeze(1) #Get central slice for VAE output
            seg_out, vae_out, mu, logvar = model(inp_imgs)
            loss = criterion(seg_out, mask, vae_out, central_slice, mu, logvar)
        elif params.net == "UNET_2D":
            seg_out = model(inp_imgs)
            loss = criterion(seg_out, mask)
        elif params.net in ["REF", "MOD_01", "MOD_02", "MOD_03"]:
            seg_pred, rec_pred, y_mid = model(inp_imgs)
            loss = criterion(seg_pred, mask, rec_pred, out_imgs, y_mid)

        loss.backward()
        optimizer.step()

        train_bar.set_postfix(loss=loss.item())


    #---Validation    
    if(params.validation):
        validation_metrics[epoch,:] = test_model(model, val_loader, params.net, VAE_enable = params.VAE_enable)
        np.save(results_path / "training_metrics.npy", validation_metrics)
        dice = validation_metrics[epoch,0]
        best_epoch = dice > best_val_dice
        if(best_epoch): best_val_dice = dice

     #---Logging 
    if(best_epoch and params.save_model_each_epoch):
        checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, results_path / "checkpoint.pth.tar")
        
        plot_examples(model, val_dataset, range(10), results_path, params.net, VAE_enable = params.VAE_enable)


    scheduler.step() #Adjust learning rate
    
    
