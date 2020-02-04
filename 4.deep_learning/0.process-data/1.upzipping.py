import pandas as pd
import sys
from pathlib import Path
import os

basefolder_loc = Path(__file__).parents[2]

sys.path.append(str(basefolder_loc))
from utils import get_all_images

csv_data_dir = os.path.join(basefolder_loc, "2.process-data", "data")

output_loc = os.path.join("data", "train")
file_loc = os.path.join(csv_data_dir, "train_processed.tsv.gz")
data_train_csv = pd.read_csv(file_loc, sep="\t")

all_images_train = get_all_images(data_train_csv)

for image_object in all_images_train:
    save_loc = os.path.join(output_loc, image_object[2])
    os.makedirs(save_loc, exist_ok=True)
    image_object[0].save(os.path.join(save_loc, image_object[1] + ".png"))
