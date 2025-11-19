'usage: python plot_times.py "file_to_save'
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

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
stats = f"Avg:{avg:.1f} secs\nMax:{max:.1f} secs\nMin:{min:.1f} secs"

plt.figure()
plt.plot(epochs, epoch_times_to_plot)
plt.xlabel("Epoch")
plt.ylabel("Time per epoch (seconds)")
plt.title(f"Training time per epoch: {sys.argv[1]}")
plt.annotate(stats, (0.01,0.85),xycoords='axes fraction')
plt.tight_layout()
plt.savefig(save_path)
#plt.show()
