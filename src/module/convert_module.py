import re
import json
from ..convert import uai, nips, ICLR_Formatter

def process_uai_data(p_r):
    p, r = p_r
    paper = extract_single_pdf(p)
    with open(r, "r") as fp:
        review = json.load(fp)
    meta = uai(review, paper)
    pattern = r"UAI/(\d{4})"
    year = int(re.findall(pattern, r)[0])
    return meta, p, year

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
    pattern = r"ICLR_(\d{4})"
    year = int(re.findall(pattern, r)[0])
    return meta, p, year

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
    pattern = r"NeurIPS/(\d{4})"
    year = int(re.findall(pattern, r)[0])
    return meta, p, year


def extract_single_pdf(single_md_path):
    with open(single_md_path, 'r') as fp:
        paper = fp.read()
    content = re.split(r'(?i)(introd)', paper, maxsplit=1)[1:]
    content = "".join(content)
    return content

def save_file(meta, index):
    with open(fr"./data/converted/ICLR/{index:05}.json", 'w') as fp:
        json.dump(meta, fp)

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