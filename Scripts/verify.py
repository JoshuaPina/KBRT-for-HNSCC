from pathlib import Path
import cv2
from tqdm import tqdm
from tabulate import tabulate

parent_dir = Path("Data/data_images")
subfolders = [f for f in parent_dir.iterdir() if f.is_dir()]


total_images = 0
image_sizes = set()

for folder in tqdm(subfolders, desc="Scanning folders"):
    image_files = list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
    
    folder_image_count = len(image_files)
    total_images += folder_image_count
    
    print(f"{folder.name}: {folder_image_count} image(s)")

for img_path in tqdm(image_files, desc=f"🔍 {folder.name}", leave=False):
    try:
        img = cv2.imread(str(img_path))
        if img is not None:
            image_sizes.add(img.shape[:2])
    except Exception as e:
        print(f"Error reading {img_path.name}: {e}")
print()
print("-"*80)
print()
if total_images >= 20000:
    print("I feel like you're just here for the zipline...")
else:
    print("Oh...I thought you were here for the data verification.")
print()
print("-"*80)
print()

#data for my table
summary_table = [
    ["Total Folders", len(subfolders)],
    ["Total Images", total_images],
    ["Unique Image Sizes", len(image_sizes)],
    ["Image Size", ', '.join(str(size) for size in image_sizes)],
    ["Data Verified?", "Yes" if (
        total_images == 91135 and 
        len(subfolders) == 335 and 
        image_sizes == {(128, 128)}
    ) else "No"]
]

# Print summary using tabulate
print("Summary of Data Verification:")
print(tabulate(summary_table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

print()
if total_images == 91135 and len(subfolders) == 335 and image_sizes == {(128, 128)}:
    print("You did it! You executed Order 66!")
print("\nThis is the end of the data verification. May the force be with you, always.")
print()
print("-"*80)
