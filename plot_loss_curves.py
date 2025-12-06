import sys
import pickle
from pathlib import Path
import numpy as np
import torch
from src.testing_functions import plot_loss_curves

folder = sys.argv[1]

#Directory for output results
results_path = Path('training_results')/folder

with open(results_path/'params.pkl', 'rb') as f:
    params = pickle.load(f) 
validation_metrics = np.load(results_path / "validation_metrics.npy")
training_metrics = np.load(results_path / "training_metrics.npy")
checkpoint = torch.load(results_path/"checkpoint.pth.tar", weights_only = False, map_location=torch.device('cpu')) 
epoch = checkpoint['epoch']

plot_loss_curves(results_path, validation_metrics, training_metrics, epoch, params.VAE_enable, True, params.net)
