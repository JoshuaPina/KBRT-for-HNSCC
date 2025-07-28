import os
import torch
from torchvision.io import read_image
from torchvision.transforms import Resize, ToPILImage
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from PIL import Image
import numpy as np

# Takes average of all a patient's CT files, outputs to a single jpg for each patient.

INPUT_ROOT = "../HNSCC_data/ct_images/"
OUTPUT_ROOT = "ct_images_avg/"
LOG_FILE = "ct_slice_counts.txt"  # Logs amount of slices used for each patient, just so we know.
OUTPUT_SIZE = (128, 128)

os.makedirs(OUTPUT_ROOT, exist_ok=True)
resize = Resize(OUTPUT_SIZE)
to_pil = ToPILImage()

log_lines = []

for folder in tqdm(os.listdir(INPUT_ROOT)):
    if not folder.endswith("_ct_images"):
        continue

    folder_path = os.path.join(INPUT_ROOT, folder)
    slice_paths = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".png")
        ]
    )

    if not slice_paths:
        continue

    volume = []
    for path in slice_paths:
        img = Image.open(path).convert("RGB")
        if img.size != OUTPUT_SIZE:
            img = resize(img)
        volume.append(to_tensor(img))

    avg_tensor = torch.stack(volume).mean(dim=0)
    avg_image = to_pil(avg_tensor)

    out_id = folder.replace("_ct_images", "")
    out_path = os.path.join(OUTPUT_ROOT, f"{out_id}.jpg")
    avg_image.save(out_path)

    log_lines.append(f"{out_id}: {len(slice_paths)} slices")

# Save slice count log
with open(LOG_FILE, "w") as logf:
    logf.write("\n".join(log_lines))

print("Averaged images saved to:", OUTPUT_ROOT)
print("Log saved to:", LOG_FILE)
