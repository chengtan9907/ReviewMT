import requests
import re
import os
from tqdm import tqdm
import argparse
import json
import yaml

headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
}
parser = argparse.ArgumentParser(description='Get scripts for iclr')

def download_pdf(outpath1, outpath2, number, aper_link, peer_review_link, is_peer_review):
    path1 = os.path.join(outpath1, f'{number:04d} paper.pdf')
    path2 = os.path.join(outpath2, f'{number:04d} peer_review.pdf')
    while True:
        try:
            response1 = requests.get(paper_link, headers=headers)
        except requests.exceptions.RequestException:
            print(f"number:{number}occured an error, retring...")
            continue
        if '200' in str(response1):
            break
        print("Requests error, retrying...")
    if is_peer_review:
        while True:
            try:
                response2 = requests.get(peer_review_link, headers=headers)
            except:
                print(f"number:{number}occured an error, retring...")
                continue
            if '200' in str(response2):
                break
            print("Requests error, retrying...")
    with open(path1, 'wb') as f:
        f.write(response1.content)
    if is_peer_review:
        with open(path2, 'wb') as f:
            f.write(response2.content)

def make_t_a(raw_string, outpath, number):
    pattern = r'<time datetime="\d{4}-(\d{2})-\d{2}">'
    month = int(re.findall(pattern, raw_string)[0])
    print(month)
    pattern = r'<div class="c-article-section__content" id="Abs1-content"><p>(.*?)</p></div>'
    abst = re.findall(pattern, raw_string)
    ta_ok = True
    if len(abst) != 1:
        print(f"{number}'s abstract lost")
        ta_ok = False
    else:
        abst = abst[0]
    pattern = r'<h1 class="c-article-title" data-test="article-title" data-article-title="">(.*?)</h1>'
    title = re.findall(pattern, raw_string)
    if len(title) != 1:
        print(f"{number}'s title lost")
        ta_ok = False
    else:
        title = title[0]
    meta = {
        'title': title,
        'abstract': abst,
        'month': month
    }
    if ta_ok:
        with open(os.path.join(outpath,f"{number:04d} nature_ta.json"), 'w', encoding='utf-8') as fp:
            json.dump(meta, fp)

number = 1
is_peer_review = True

parser.add_argument("path2yaml", type=str, help="Path to the arguments yaml")
yaml_path = parser.parse_args().path2yaml

if not os.path.exists(yaml_path):
    print("The path of yaml is not exists")
    raise Exception("path not exists")


with open(yaml_path, 'r') as fp:
    y = yaml.safe_load(fp)

if 'outpath1' in y:
    outpath1 = y['outpath1']
else:
    raise Exception("not enough arguments")
if 'outpath2' in y:
    outpath2 = y['outpath2']
else:
    raise Exception("not enough arguments")
if 'outpath3' in y:
    outpath3 = y['outpath3']
else:
    raise Exception("not enough arguments")
if 'proxy' in y:
    os.environ['https_proxy'] = y['proxy']

for page_number in tqdm(range(1, 400), position=0, desc = "Processing:", leave=False):
    url = f'https://www.nature.com/ncomms/research-articles?searchType=journalSearch&sort=PubDate&type=article&year=2023&page={page_number}'
    while True:
        try:
            resp = requests.get(url = url, headers=headers)
        except requests.exceptions.RequestException:
                print(f"number:{number}occured an error, retring...")
                continue
        if '200' in str(resp):
            break
        print("Requests error, retrying...")
    string = str(resp.content)
    while True:
        try:
            resp_page = requests.get(url = url, headers=headers)
        except requests.exceptions.RequestException:
                print(f"number:{number}occured an error, retring...")
                continue
        if '200' in str(resp):
            break
        print("Requests error, retrying...")
    pattern = r'<a href="/articles/(.*?)"'
    mathches = re.findall(pattern, string)
    for a_paper_title in tqdm(mathches, position=1, leave=False):
        url2 = f'https://www.nature.com/articles/{a_paper_title}'
        while True:
            try:
                resp_paper = requests.get(url = url2, headers=headers)
            except requests.exceptions.RequestException:
                print(f"number:{number}occured an error, retring...")
                continue
            if '200' in str(resp_paper):
                break
            print("Requests error, retrying...")
        string2 = str(resp_paper.content)
        make_t_a(string2, outpath3, number)
        pattern_peer_review = r'data-track-label="peer review file" href="(.*?)"'
        mathches2 = re.findall(pattern_peer_review, string2)
        pattern_paper_title = r'<h1 class="c-article-title".*data-article-title="">(.*?)</h1>'
        paper_link = url2 + '.pdf'
        if len(mathches2) == 1:
            peer_review_link = mathches2[0]
        else:
            if len(mathches2) == 0:
                print(f"\n{number} don't have peer_review\n")
                is_peer_review = False
            else:
                print(f'{number} unkown error')
                break
        
        download_pdf(outpath1, outpath2, number, paper_link, peer_review_link, is_peer_review)
        is_peer_review = True
        number += 1