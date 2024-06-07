import yaml
import os
import argparse
from module import clean

parser = argparse.ArgumentParser(description='Convert scripts for nature')

parser.add_argument("path2yaml", type=str, help="Path to the arguments yaml")
yaml_path = parser.parse_args().path2yaml

if not os.path.exists(yaml_path):
    print("The path of yaml is not exists")
    raise Exception("path not exists")
with open(yaml_path, 'r') as fp:
    y = yaml.safe_load(fp)

arg = ""

if 'marker_path' in y:
    marker_path = y['marker_path']
else:
    raise Exception("not enough arguments")
if 'inpath1' in y:
    inpath1 = y['inpath1']
else:
    raise Exception("not enough arguments")
if 'inpath2' in y:
    inpath2 = y['inpath2']
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
if 'workers' in y:
    workers = y['workers']
    if not arg:
        arg += " "
    arg += f"--workers {workers} "
if 'min_length' in y:
    min_length = y['min_length']
    if not arg:
        arg += " "
    arg += f"--min_length {min_length} "
if 'max' in y:
    mmax = y['max']
    if not arg:
        arg += " "
    arg += f"--max {mmax} "

print("Start Converting Process(nature)")

os.system(f"cd {marker_path}")
os.system(f"marker {inpath1} {outpath1} {arg}")
os.system(f"marker {inpath2} {outpath2} {arg}")

choose = input("Do you want to keep the images and meta(json) files extracted together? (This content will not be used later in this project) (n/Y)")

if not choose == "Y":
    clean(outpath1)
    clean(outpath2)
    print("All files have been cleaned.")

print("Finish Converting Process(nature)")