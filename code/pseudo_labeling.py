from pathlib import Path

import numpy as np
import pandas as pd
import torch
from segmentation_models_pytorch.utils import metrics
from tqdm import tqdm
from dataset import ChipsDataset
from models import SimpleSegNet

chpt_path = "logs/testing.pth"
model = SimpleSegNet(4, 1)
model.load_state_dict(torch.load(chpt_path))
model.eval()
metric = metrics.IoU(threshold=0.5)

df_inference = pd.read_csv("../notebooks/chip_index.csv")
# dataset = ChipsDataset(df_inference)
df = pd.DataFrame()

p_bar = tqdm(df_inference.iterrows())
for i, row in p_bar:
    img, mask = row["img_path"], row["mask_path"]
    img, mask = np.load(img), np.load(mask)
    img = torch.from_numpy(img/255).float()
    mask = torch.from_numpy(mask/255).float()
    img, mask = img.unsqueeze(0), mask.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)
    iou = metric(mask, pred)
    # print(iou)
    # exit()

    p_bar.set_postfix({"iou": iou.item()})
    d = {
        "img_path": row["img_path"],
        "mask_path": row["mask_path"],
        "iou": iou.item()
    }
    df = df.append(d, ignore_index=True)

df.to_csv("pseudo_label_scores.csv", index=False)
