
import os, json, random
import numpy as np
import torch
from datetime import datetime

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def now_ts():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def save_results(path_dir, payload: dict, prefix: str):
    os.makedirs(path_dir, exist_ok=True)
    ts = now_ts()
    json_path = os.path.join(path_dir, f"{prefix}_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    return json_path
