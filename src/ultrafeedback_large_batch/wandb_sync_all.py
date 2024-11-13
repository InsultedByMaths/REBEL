import glob
import subprocess


for r in glob.glob("./wandb/offline-*"):
    print("wandb", "sync", r)
    subprocess.run(["wandb", "sync", r])