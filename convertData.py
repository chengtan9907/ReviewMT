from src.convert import uai, nips, ICLR_Formatter
import glob
from tqdm import tqdm
import json
import os
import re
import json
import torch.multiprocessing as mp

def find_name(raw_list, name):
    for i in raw_list:
        if name in i:
            return raw_list.index(i)
    return False

def extract_single_pdf(single_md_path):
    global BATCH_PROCESS_IN_ADVANCE
    if BATCH_PROCESS_IN_ADVANCE:
        with open(single_md_path, 'r') as fp:
            paper = fp.read()
        content = re.split(r'(?i)(introd)', paper, maxsplit=1)[1:]
        content = "".join(content)
        return content
        
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
    # print(match_pdf_and_review(nips_paper_path, nips_review_path))
    
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
    # print(match_pdf_and_review(nips_paper_path, nips_review_path))
    return nips_review_path, nips_paper_path, iclr_review_path, iclr_paper_path, uai_review_path, uai_paper_path
    
def match_pdf_and_review(pdf, review):
    pdf = [i.split(".")[1] for i in pdf]
    pdf.sort()
    review = [i.split(".")[1] for i in review]
    review.sort()
    
    for (p, r) in zip(pdf, review):
        # print(f"{p} {r}")
        if p != r:
            return False
    return True

def fix_nips(review):
    for r in review['reviewers']:
        content = json.loads(r['summary'])
        r['summary'] = " ".join([content['summary_and_contributions'], content['opportunities_for_improvement'], content['limitations']])
        r['rating'] = content['rating']
        r['confidence'] = content['confidence']
    return review
        
def convert_nips(nips_review_path, nips_paper_path):
    # convert NIPS data
    print("Start to convert NIPS data.")
    os.makedirs(r"./data/converted/NeurIPS", exist_ok=True)
    for index, (p, r) in tqdm(enumerate(zip(nips_paper_path, nips_review_path)), position=0, leave=False, total=len(nips_paper_path)):
        paper = extract_single_pdf(p)
        with open(r, "r") as fp:
            review = json.load(fp)
        if '2023_track' in r:
            review = fix_nips(review)
        meta = nips(review, paper)
        with open(fr"./data/converted/NeurIPS/{index:04}.json", 'w') as fp:
            json.dump(meta, fp)

def convert_iclr(iclr_review_path, iclr_paper_path):
    # convert ICLR data
    print("Start to convert ICLR data.")
    iclr = ICLR_Formatter()
    os.makedirs(r"./data/converted/ICLR", exist_ok=True)
    iclr_review_path.sort()
    iclr_paper_path.sort()
    for index, (p, r) in tqdm(enumerate(zip(iclr_paper_path, iclr_review_path)), position=0, leave=False, total=len(iclr_paper_path)):
        year = p.split("/")[-1][5:9]
        method_name = f"deal{year}"
        corresponding_method = getattr(iclr, method_name)
        paper = extract_single_pdf(p)
        with open(r, "r") as fp:
            review = json.load(fp)
        iclr.base(review, paper)
        meta = corresponding_method(review['reviewers'])
        with open(fr"./data/converted/ICLR/{index:05}.json", 'w') as fp:
            json.dump(meta, fp)

def convert_uai(uai_review_path, uai_paper_path):
    # convert UAI data
    print("Start to convert UAI data.")
    os.makedirs(r"./data/converted/UAI", exist_ok=True)
    for index, (p, r) in tqdm(enumerate(zip(uai_paper_path, uai_review_path)), position=0, leave=False, total=len(uai_paper_path)):
        paper = extract_single_pdf(p)
        with open(r, "r") as fp:
            review = json.load(fp)
        meta = uai(review, paper)
        with open(fr"./data/converted/UAI/{index:03}.json", 'w') as fp:
            json.dump(meta, fp)
    print("All convertion have been done.")

def main():
    try:
        mp.set_start_method('spawn') # Required for CUDA, forkserver doesn't work
    except RuntimeError:
        raise RuntimeError("Set start method to spawn twice. This may be a temporary issue with the script. Please try running it again.")
    
    nips_review_path, nips_paper_path, iclr_review_path, iclr_paper_path, uai_review_path, uai_paper_path = read_all_data()
    INPUT = input("Is prepared data correct? [y/n]")
    if INPUT == 'n' or INPUT == 'N':
        exit()
    print("Start to convert all dataset")
    convert_nips(nips_review_path, nips_paper_path)
    
    convert_iclr(iclr_review_path, iclr_paper_path)
    
    convert_uai(uai_review_path, uai_paper_path)
    print("Convert successfully!")
if __name__ == '__main__':
    global BATCH_PROCESS_IN_ADVANCE
    BATCH_PROCESS_IN_ADVANCE = True
    main()