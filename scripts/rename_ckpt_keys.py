# TODO: DELETE THIS FILE

import torch

ckpt_path = "/checkpoint/sam3/haithamkhedr/checkpoints/sam3_dense/sam3_v4.pt"

ckpt = torch.load(ckpt_path, map_location="cpu")
sd = ckpt["model"]

new_sd = {}
for key, value in sd.items():
    if key.startswith("sam3_model."):
        new_key = key.replace("sam3_model.", "detector.")
        new_sd[new_key] = value
    elif key.startswith("sam2_predictor.model."):
        new_key = key.replace("sam2_predictor.model.", "tracker.")
        new_sd[new_key] = value
    else:
        new_sd[key] = value

torch.save(new_sd, "assets/sam3.pt")
