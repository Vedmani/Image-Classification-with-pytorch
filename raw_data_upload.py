import wandb
import os
import secret_keys
import argparse
from pathlib import Path

wandb_api = secret_keys.wandb_api
wandb.login(key=wandb_api)

parser = argparse.ArgumentParser()
parser.add_argument("--raw_data_path", type=str, default="data", required=True, help="path to raw data")
args = parser.parse_args()

print(args.raw_data_path)
SRC = args.raw_data_path

run = wandb.init(project="indian-birds", job_type="upload")
art = wandb.Artifact("raw_data", type="dataset")
labels = os.listdir(SRC)
for label in labels:
    print(SRC+"/"+label)
    art.add_dir(SRC+"/"+label, name=label)
run.log_artifact(art)
run.finish()
