from pathlib import Path
import os
import torch

label = "custom_vae_2d_test"

BASE_DIR = Path(__file__).resolve().parent
results_path = BASE_DIR / "training_results" / label
checkpoint_path = results_path / "checkpoint.pth.tar"

print("CWD:", os.getcwd())
print("BASE_DIR:", BASE_DIR)
print("results_path:", results_path, "exists:", results_path.exists())
print("checkpoint_path:", checkpoint_path, "exists:", checkpoint_path.exists())
if results_path.exists():
    print("Contents of results_path:")
    for p in results_path.iterdir():
        print("  ", p.name)

if checkpoint_path.exists():
    print("\nTrying to load checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    print("Loaded keys:", ckpt.keys())

