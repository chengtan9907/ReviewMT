import argparse
import importlib
from tqdm import tqdm
import yaml
import os

parser = argparse.ArgumentParser(description='Get scripts for iclr')

parser.add_argument("path2yaml", type=str, help="Path to the arguments yaml")
yaml_path = parser.parse_args().path2yaml

if not os.path.exists(yaml_path):
    print("The path of yaml is not exists")
    raise Exception("path not exists")
with open(yaml_path, 'r') as fp:
    y = yaml.safe_load(fp)

if 'years' in y:
    years = y['years']
else:
    raise Exception("not enough arguments")

if 'outpath1' in y:
    outpath1 = y['outpath1']
else:
    raise Exception("not enough arguments")
if 'outpath2' in y:
    outpath2 = y['outpath2']
else:
    raise Exception("not enough arguments")
if 'proxy' in y:
    os.environ['https_proxy'] = y['proxy']

print("Start to download ICLR raw data")
for year in tqdm(years, position=0, desc = "Processing:", leave=False):
    if not(year>=2017 and year<=2024):
        raise Exception(f"Do not have data about {year}, please input in range 2017 to 2024")
    webcrawler = importlib.import_module(f"iclr.ICLR{year}")
    webcrawler.get_paper_and_review(outpath1, outpath2)

print("All ICLR raw data have been downloaded.")