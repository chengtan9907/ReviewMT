import json
import re
import os
from tqdm import tqdm
import argparse
import yaml
from module import make_iclr

parser = argparse.ArgumentParser(description='Make datasets for iclr')

parser.add_argument("path2yaml", type=str, help="Path to the arguments yaml")
yaml_path = parser.parse_args().path2yaml

if not os.path.exists(yaml_path):
    print("The path of yaml is not exists")
    raise Exception("path not exists")
with open(yaml_path, 'r') as fp:
    y = yaml.safe_load(fp)

if 'inpath1' in y:
    paper_path = y['inpath1']
else:
    raise Exception("not enough arguments")
if 'inpath2' in y:
    review_path = y['inpath2']
else:
    raise Exception("not enough arguments")
if 'outpath' in y:
    outpath = y['outpath']
else:
    raise Exception("not enough arguments")

print("Start to make ICLR datasets:")

dirs = os.listdir(paper_path)
dirs.sort()
datasets = [[] for _ in range(8)]
for d in tqdm(dirs, position=0, desc = "Processing:", leave=False):
    pattern = r'ICLR_(\d{4})_paper_(\d{4})\.md'
    match = re.search(pattern, str(d))
    year = match.group(1)
    name = match.group(2)
    if os.path.exists(os.path.join(review_path,f"ICLR_{year}_review_{name}.json")):
        review_fpath = os.path.join(review_path,f"ICLR_{year}_review_{name}.json")
    else:
        print(f"{review_path}/ICLR_{year}_review_{name}.json file does not exists")
        continue
    meta = make_iclr(fr"{paper_path}/ICLR_{year}_paper_{name}.md", review_fpath)
    datasets[int(year)-2017].append(meta)
for index in range(8):
    year = index+2017
    if datasets[index]:
        with open(os.path.join(outpath,f'ICLR{year}_datasets.json'), 'w', encoding='utf-8') as fp:
            json.dump(datasets[index], fp)

print("All datasets has made, process done.")