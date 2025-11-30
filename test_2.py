import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Usage: python plot_time_metrics.py <folder_name>
results_path = Path("training_results") / sys.argv[1]

# ---------- Load epoch times (existing behavior) ----------
epoch_times_path = results_path / "epoch_times.npy"
epoch_times = np.load(epoch_times_path)

# Remove trailing zeros (unfinished epochs)
nonzero_idxs = np.nonzero(epoch_times)[0]
if len(nonzero_idxs) == 0:
    raise ValueError("epoch_times.npy contains only zeros – nothing to plot.")
last_nonzero_idx = nonzero_idxs[-1]

epoch_times_to_plot = epoch_times[:last_nonzero_idx + 1]
epochs = np.arange(1, len(epoch_times_to_plot) + 1)

# ---------- Load new metrics if available ----------
train_times = None
val_times = None
gpu_mem_usage = None
model_size_mb = None

train_times_path = results_path / "train_times.npy"
val_times_path = results_path / "val_times.npy"
gpu_mem_path = results_path / "gpu_mem_usage.npy"
model_size_txt = results_path / "model_size_mb.txt"

if train_times_path.exists():
    arr = np.load(train_times_path)
    train_times = arr[:last_nonzero_idx + 1]

if val_times_path.exists():
    arr = np.load(val_times_path)
    val_times = arr[:last_nonzero_idx + 1]

if gpu_mem_path.exists():
    arr = np.load(gpu_mem_path)
    gpu_mem_usage = arr[:last_nonzero_idx + 1]

if model_size_txt.exists():
    try:
        with open(model_size_txt, "r") as f:
            model_size_mb = float(f.readline().strip())
    except Exception:
        model_size_mb = None

# ==========================================================
# 1) ORIGINAL PLOT: total time per epoch
# ==========================================================

avg = np.mean(epoch_times_to_plot)
max_time = np.max(epoch_times_to_plot)
min_time = np.min(epoch_times_to_plot)
std_err = np.std(epoch_times_to_plot, ddof=1) / np.sqrt(len(epoch_times_to_plot))

stats = (
    f"Avg: {avg:.1f} s\n"
    f"σ/√N: {std_err:.2f} s\n"
    f"Max: {max_time:.1f} s\n"
    f"Min: {min_time:.1f} s"
)

save_path = results_path / "epoch_times_plot.png"

plt.figure()
plt.plot(epochs, epoch_times_to_plot, label="Total epoch time")
plt.xlabel("Epoch")
plt.ylabel("Time per epoch (seconds)")
plt.title(f"Training time per epoch: {sys.argv[1]}")
plt.annotate(stats, (0.01, 0.85), xycoords="axes fraction")
plt.legend()
plt.tight_layout()
plt.savefig(save_path)
plt.close()

# ==========================================================
# 2) NEW PLOT: train vs validation vs total epoch time
# ==========================================================

if train_times is not None or val_times is not None:
    save_path_tv = results_path / "train_val_times_plot.png"

    plt.figure()
    plt.plot(epochs, epoch_times_to_plot, label="Total epoch time", linewidth=2)

    if train_times is not None:
        plt.plot(epochs, train_times, "--", label="Train time")

    if val_times is not None:
        plt.plot(epochs, val_times, "--", label="Validation time")

    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.title(f"Train / Validation / Total time per epoch: {sys.argv[1]}")

    # Simple text stats for train/val if available
    text_lines = []
    if train_times is not None:
        text_lines.append(
            f"Train avg: {np.mean(train_times):.1f} s\n"
            f"Train max: {np.max(train_times):.1f} s"
        )
    if val_times is not None:
        text_lines.append(
            f"Val avg: {np.mean(val_times):.1f} s\n"
            f"Val max: {np.max(val_times):.1f} s"
        )

    if text_lines:
        plt.annotate(
            "\n\n".join(text_lines),
            (0.01, 0.75),
            xycoords="axes fraction"
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path_tv)
    plt.close()

# ==========================================================
# 3) NEW PLOT: GPU peak memory usage per epoch
# ==========================================================

if gpu_mem_usage is not None:
    save_path_gpu = results_path / "gpu_mem_usage_plot.png"

    plt.figure()
    plt.plot(epochs, gpu_mem_usage, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Peak GPU memory (GB)")
    plt.title(f"GPU peak memory usage per epoch: {sys.argv[1]}")

    # Stats + optional model size as text label
    gpu_avg = np.mean(gpu_mem_usage)
    gpu_max = np.max(gpu_mem_usage)
    gpu_min = np.min(gpu_mem_usage)

    gpu_stats = (
        f"Avg: {gpu_avg:.2f} GB\n"
        f"Max: {gpu_max:.2f} GB\n"
        f"Min: {gpu_min:.2f} GB"
    )

    if model_size_mb is not None:
        gpu_stats += f"\nModel size: {model_size_mb:.1f} MB"

    plt.annotate(gpu_stats, (0.01, 0.75), xycoords="axes fraction")

    plt.tight_layout()
    plt.savefig(save_path_gpu)
    plt.close()
