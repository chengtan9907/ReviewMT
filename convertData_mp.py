import glob
import json
import os
import re
import torch.multiprocessing as mp
from tqdm import tqdm
from src.convert import uai, nips, ICLR_Formatter


def find_name(raw_list, name):
    for i in raw_list:
        if name in i:
            return raw_list.index(i)
    return False


def extract_single_pdf(single_md_path):
    with open(single_md_path, 'r') as fp:
        paper = fp.read()
    content = re.split(r'(?i)(introd)', paper, maxsplit=1)[1:]
    content = "".join(content)
    return content

def parse_review_string(review_string):
    keys = ["summary_and_contributions", "opportunities_for_improvement", "limitations", "rating", "confidence"]
    pattern = re.compile(r'(\b\w+:\s\{\s*\'value\'\s*:\s*\'(.*?)\'\s*\})')
    matches = pattern.findall(review_string)
    result = {}
    for key, value in matches:
        clean_key = key.split(':')[0].strip()
        if clean_key in keys:
            result[clean_key] = value.strip()
    return result

def fix_nips(review):
    for r in review['reviewers']:
        content = parse_review_string(r['summary'])
        try:
            r['summary'] = " ".join([content['summary_and_contributions'] if 'summary_and_contributions' in content else "",
                                      content['opportunities_for_improvement'] if 'opportunities_for_improvement' in content else "",
                                        content['limitations'] if 'limitations' in content else ""])
        except Exception:
            print(content)
            input("continue")
        r['rating'] = content['rating']
        if not 'confidence' in content:
            return False
        r['confidence'] = content['confidence']
    return review

def process_nips_data(p_r):
    p, r = p_r
    paper = extract_single_pdf(p)
    with open(r, "r") as fp:
        review = json.load(fp)
    if '2023_track' in r:
        review = fix_nips(review)
        if review == False:
            return False
    meta = nips(review, paper)
    return meta, p

def convert_nips(nips_review_path, nips_paper_path):
    print("Start to convert NIPS data.")
    os.makedirs(r"./data/converted/NeurIPS", exist_ok=True)
    pool = mp.Pool(mp.cpu_count())

    # tqdm setup
    results = []
    with tqdm(total=len(nips_paper_path), desc="NIPS Conversion", position=0) as pbar:
        for result in pool.imap_unordered(process_nips_data, zip(nips_paper_path, nips_review_path)):
            if result == False:
                continue
            results.append(result)
            pbar.update()

    pool.close()
    pool.join()

    for index, (meta, p) in enumerate(results):
        with open(fr"./data/converted/NeurIPS/{index:04}.json", 'w') as fp:
            json.dump(meta, fp)


def process_iclr_data(p_r):
    p, r = p_r
    year = p.split("/")[-1][5:9]
    method_name = f"deal{year}"
    iclr = ICLR_Formatter()
    corresponding_method = getattr(iclr, method_name)
    paper = extract_single_pdf(p)
    with open(r, "r") as fp:
        review = json.load(fp)
    iclr.base(review, paper)
    meta = corresponding_method(review['reviewers'])
    return meta, p


def convert_iclr(iclr_review_path, iclr_paper_path):
    print("Start to convert ICLR data.")
    os.makedirs(r"./data/converted/ICLR", exist_ok=True)
    pool = mp.Pool(mp.cpu_count())

    # tqdm setup
    results = []
    with tqdm(total=len(iclr_paper_path), desc="ICLR Conversion", position=1) as pbar:
        for result in pool.imap_unordered(process_iclr_data, zip(iclr_paper_path, iclr_review_path)):
            results.append(result)
            pbar.update()

    pool.close()
    pool.join()

    for index, (meta, p) in enumerate(results):
        with open(fr"./data/converted/ICLR/{index:05}.json", 'w') as fp:
            json.dump(meta, fp)


def process_uai_data(p_r):
    p, r = p_r
    paper = extract_single_pdf(p)
    with open(r, "r") as fp:
        review = json.load(fp)
    try:
        meta = uai(review, paper)
    except Exception:
        print(r)
        input("continue")
    return meta, p


def convert_uai(uai_review_path, uai_paper_path):
    print("Start to convert UAI data.")
    os.makedirs(r"./data/converted/UAI", exist_ok=True)
    pool = mp.Pool(mp.cpu_count())
    
    # tqdm setup
    results = []
    with tqdm(total=len(uai_paper_path), desc="UAI Conversion", position=2) as pbar:
        for result in pool.imap_unordered(process_uai_data, zip(uai_paper_path, uai_review_path)):
            results.append(result)
            pbar.update()

    pool.close()
    pool.join()

    for index, (meta, p) in enumerate(results):
        with open(fr"./data/converted/UAI/{index:03}.json", 'w') as fp:
            json.dump(meta, fp)


def read_all_data():
    # read NIPS data
    raw_paper_path = glob.glob(r"./data/tmp/NeurIPS/**/**.md")
    raw_paper_path.sort()
    raw_review_path = glob.glob(r"./data/NeurIPS/**/**/**.json")
    raw_review_path = [i for i in raw_review_path if not 'content' in i]
    nips_review_path = []
    nips_paper_path = []
    for paper in raw_paper_path:
        name = paper.split("/")[-1].replace(".md", "")
        index = find_name(raw_review_path, name)
        if index != False:
            nips_review_path.append(raw_review_path[index])
            nips_paper_path.append(paper)
    nips_review_path.sort()
    nips_paper_path.sort()
    print(f"nips_review: {len(nips_review_path)}")
    print(f"nips_paper: {len(nips_paper_path)}")

    # read ICLR data
    with open("./invalidPDF.txt", 'r') as fp:
        invalid_pdf = fp.readlines()
    invalid_pdf = [i.strip().replace("pdf", "md") for i in invalid_pdf]
    invalid_review = []
    for i in invalid_pdf:
        invalid_review.append(i.replace("md", "json"))

    raw_paper_path = glob.glob(r"./data/tmp/ICLR/**/**.md")
    raw_paper_path = [i for i in raw_paper_path if not i.split("/")[-1] in invalid_pdf]
    raw_review_path = glob.glob(r"./data/iclr_reviews/**.json")
    raw_review_path = [i for i in raw_review_path if not i.split("/")[-1] in invalid_review]
    iclr_review_path = []
    iclr_paper_path = []
    for paper in raw_paper_path:
        name = paper.split("/")[-1].replace(".md", "").replace("paper", "review")
        index = find_name(raw_review_path, name)
        if index != False:
            iclr_review_path.append(raw_review_path[index])
            iclr_paper_path.append(paper)
    iclr_review_path.sort()
    iclr_paper_path.sort()
    print(f"iclr_review: {len(iclr_review_path)}")
    print(f"iclr_paper: {len(iclr_paper_path)}")

    # read UAI data
    uai_review_path = glob.glob(r"./data/UAI/**/**/**.json")
    uai_review_path = [i for i in uai_review_path if not "_content" in i]
    uai_paper_path = glob.glob(r"./data/tmp/UAI/**/**.md")
    uai_review_path.sort()
    uai_paper_path.sort()
    print(f"uai_review: {len(uai_review_path)}")
    print(f"uai_paper: {len(uai_paper_path)}")

    return nips_review_path, nips_paper_path, iclr_review_path, iclr_paper_path, uai_review_path, uai_paper_path


def main():
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    nips_review_path, nips_paper_path, iclr_review_path, iclr_paper_path, uai_review_path, uai_paper_path = read_all_data()
    INPUT = input("Is prepared data correct? [y/n]")
    if INPUT.lower() == 'n':
        exit()
    print("Start to convert all datasets")
    convert_nips(nips_review_path, nips_paper_path)
    convert_iclr(iclr_review_path, iclr_paper_path)
    convert_uai(uai_review_path, uai_paper_path)
    print("Conversion successfully completed!")
    num1 = len(os.listdir(r"./data/converted/NeurIPS"))
    num2 = len(os.listdir(r"./data/converted/ICLR"))
    num3 = len(os.listdir(r"./data/converted/UAI"))
    print(f"NeurIPS: {num1}; ICLR: {num2}; UAI: {num3}")


if __name__ == '__main__':
    main()
