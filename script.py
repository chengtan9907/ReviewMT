import json
import glob
import re
from tqdm import tqdm

if __name__ == "__main__":
    iclr_path = glob.glob(r"data/converted/ICLR/**.json")
    iclr_path.sort()
    nips_path = glob.glob(r"data/converted/NeurIPS/**.json")
    nips_path.sort()
    path = nips_path
    accept = 0
    reject = 0
    total = 0
    
    for paper in tqdm(path):
        total += 1
        with open(paper, 'r') as fp:
            content = json.load(fp)[0]
        match = re.findall(r"Final decision: (.*?)\.", str(content['output']))
        if not match or len(match)>1:
            raise Exception("match error")
        match = match[0].lower()
        if 'accept' in match:
            accept += 1
        else:
            reject += 1
    print(f"accept/reject: {accept}/{reject}")
    print(f"total: {total}")