from src.convert import uai, nips, ICLR_Formatter
import glob
from tqdm import tqdm
import json
import os
import re
import shutil
import pypdfium2 
from marker.convert import convert_single_pdf
from marker.models import load_all_models
from marker.output import save_markdown

BATCH_MULTI = 64

def extract_pdf_content(pdf_path):
    os.makedirs(r"./data/tmp", exist_ok=True)
    model_lst = load_all_models()
    full_text, images, out_meta = convert_single_pdf(pdf_path, model_lst, max_pages=None, langs=None, batch_multiplier=BATCH_MULTI)
    pdf_path = os.path.basename(pdf_path)
    save_markdown(r"./data/tmp", pdf_path, full_text, images, out_meta)
    md_path = glob.glob(r"./data/tmp/**/**.md")[0]
    with open(md_path, 'r') as fp:
        paper = fp.read()
    content = re.split(r'(?i)(introd)', paper, maxsplit=1)[1:]
    content = "".join(content)
    shutil.rmtree(r"./data/tmp/")
    return content
    
def read_all_data():
    # read NIPS data
    raw_path = glob.glob(r"./data/NeurIPS/**/**/**.pdf")
    raw_path.sort()
    nips_review_path = []
    nips_paper_path = []
    for paper in tqdm(raw_path):
        if os.path.exists(paper.replace(".pdf", "")+".json"):
            nips_review_path.append(paper.replace(".pdf", "")+".json")
            nips_paper_path.append(paper)
    print(f"nips_review: {len(nips_review_path)}")
    print(f"nips_paper: {len(nips_paper_path)}")
    # print(match_pdf_and_review(nips_paper_path, nips_review_path))
    
    # read ICLR data
    with open("./invalidPDF.txt", 'r') as fp:
        invalid_pdf = fp.readlines()
    invalid_review = []
    for i in invalid_pdf:
        invalid_review.append(i.replace("pdf", "json"))
    
    iclr_paper_path = glob.glob(r"./data/iclr_papers/**.pdf")
    iclr_paper_path = [i for i in iclr_paper_path if not i in invalid_pdf]
    iclr_review_path = glob.glob(r"./data/iclr_reviews/**.json")
    iclr_review_path = [i for i in iclr_review_path if not i in invalid_review]
    print(f"iclr_review: {len(iclr_review_path)}")
    print(f"iclr_paper: {len(iclr_paper_path)}")
    
    # read UAI data
    uai_review_path = glob.glob(r"./data/UAI/**/**/**.json")
    uai_review_path = [i for i in uai_review_path if not "_content" in i]
    uai_review_path.sort()
    uai_paper_path = glob.glob(r"./data/UAI/**/**/**.pdf")
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
        
    # for i in pdf:
    #     if not i in review:


def main(): 
    nips_review_path, nips_paper_path, iclr_review_path, iclr_paper_path, uai_review_path, uai_paper_path = read_all_data()

    # convert NIPS data
    print("Start to convert NIPS data.")
    os.makedirs(r"./data/converted/NeurIPS", exist_ok=True)
    for index, (p, r) in tqdm(enumerate(zip(nips_paper_path, nips_review_path)), position=0, leave=False, total=len(nips_paper_path)):
        paper = extract_pdf_content(p)
        with open(r, "r") as fp:
            review = json.load(fp)
        meta = nips(review, paper)
        with open(fr"./data/converted/NeurIPS/{index:04}.json", 'w') as fp:
            json.dump(meta, fp)
    
    # convert ICLR data
    print("Start to convert ICLR data.")
    iclr = ICLR_Formatter()
    os.makedirs(r"./data/converted/ICLR", exist_ok=True)
    for index, (p, r) in tqdm(enumerate(zip(iclr_paper_path, iclr_review_path)), position=0, leave=False, total=len(iclr_paper_path)):
        year = p.split("/")[-1][5:9]
        method_name = f"deal{year}"
        corresponding_method = getattr(iclr, method_name)
        paper = extract_pdf_content(p)
        with open(r, "r") as fp:
            review = json.load(fp)
        iclr.base(review, paper)
        meta = corresponding_method(review['reviewers'])
        with open(fr"./data/converted/ICLR/{index:05}.json", 'w') as fp:
            json.dump(meta, fp)
        
    # convert UAI data
    print("Start to convert UAI data.")
    os.makedirs(r"./data/converted/UAI", exist_ok=True)
    for index, (p, r) in tqdm(enumerate(zip(uai_paper_path, uai_review_path)), position=0, leave=False, total=len(uai_paper_path)):
        paper = extract_pdf_content(p)
        with open(r, "r") as fp:
            review = json.load(fp)
        meta = uai(review, paper)
        with open(fr"./data/converted/UAI/{index:03}.json", 'w') as fp:
            json.dump(meta, fp)
    print("All convertion have been done.")
    
main()