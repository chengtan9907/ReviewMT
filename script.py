import os
import glob
import json

iclr_path = glob.glob(r"./data/converted/ICLR/**.json")
# iclr_path = glob.glob(r"./data/UAI/**.json")

iclr_path.sort()

for idx, file in enumerate(iclr_path):
    with open(file, 'r') as fp:
        content = json.load(fp)
    print(f"{idx}'s history: {len(content['history'])}")