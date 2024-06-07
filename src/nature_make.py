import json
import re
import os
from tqdm import tqdm
import argparse
import yaml
import module
import pandas as pd
import shutil

parser = argparse.ArgumentParser(description='Make datasets for nature')

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
if 'inpath3' in y:
    ta_path = y['inpath3']
else:
    raise Exception("not enough arguments")
if 'inpath4' in y:
    error_path = y['inpath4']
else:
    error_path = ""
if 'train_month' in y:
    train_month = y['train_month']
    for t in train_month:
        if not (t>=1 and t<=12):
            train_month.remove(t)
else:
    train_month = []
if 'outpath' in y:
    outpath = y['outpath']
else:
    raise Exception("not enough arguments")

print("Start to make Nature datasets:")

dirs = os.listdir(paper_path)
dirs.sort()

months = [[] for _ in range(12)]
unrecognized = []
for d in dirs:
    name = d.split('/')[-1][0:4]
    if not os.path.exists(os.path.join(review_path, f"{name} peer_review.md")):
        print(f"{name} don't have review file.")
        continue
    else:
        with open(os.path.join(review_path, f"{name} peer_review.md"), 'r', encoding='utf-8') as fp:
            review = fp.read()
        review_path2 = os.path.join(review_path, f"{name} peer_review.md")
    if not os.path.exists(os.path.join(ta_path, f"{name} nature_ta.json")):
        print(f"{name} don't have t+a file.")
        continue
    else:
        module.dealing_pdfs(paper_path, ta_path)
        with open(os.path.join(ta_path, f"{name} nature_ta.json"), 'r', encoding='utf-8') as fp:
            ta = json.load(fp)
    title = module.delete(ta['title'])
    abst = module.delete(ta['abstract'])
    content = module.delete(ta['content'])
    month = int(ta['month']) - 1
    review = module.extract_review(os.path.join(review_path, f"{name} peer_review.md"))
    if len(review)==0:
        unrecognized.append(review_path2.split('/')[-1][0:4])
    else:
        meta = module.make_nature(title, abst, content, review, "Accept")
        months[month].append(meta)

if not len(train_month)==0:
    train = []
    test = []
    for m in train_month:
        train += months[m]
    for m in range(0, 12):
        if not m in train_month:
            test += months[m]
    with open(os.path.join(outpath, "Nature_train_dataset.json"), 'w', encoding='utf-8') as fp:
        json.dump(train, fp)
    with open(os.path.join(outpath, "Nature_test_dataset.json"), 'w', encoding='utf-8') as fp:
        json.dump(test, fp)
else:
    result = []
    for m in months:
        result += m
    with open(os.path.join(outpath, "Nature_dataset.json"), 'w', encoding='utf-8') as fp:
        json.dump(result, fp)
static = {}
for m in range(0,12):
    static[m+1] = len(months[m])
static['unrecognized'] = len(unrecognized)
if not len(error_path)==0:
    for u in unrecognized:
        shutil.copy(os.path.join(review_path, f"{u} peer_review.md"), os.path.join(error_path, f"{u} peer_review.md"))
df = pd.DataFrame.from_dict(static, orient="index", columns=["value"])
df.to_csv(os.path.join(outpath, "nature_make_result.csv"), header=True)
print(r"All datasets has made, process done. Note that the statistics in the output path are saved in nature_make_result.csv")