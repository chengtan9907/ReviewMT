from src.module.readData import read_all_data
from src.convert import process_iclr
from src.module.convert_module import extract_single_pdf
import json
from tqdm import tqdm

def process_dpo_data(p, r):
    method_name = p.split("/")[-1][5:9]
    paper = extract_single_pdf(p)
    with open(r, "r") as fp:
        review = json.load(fp)
    meta = process_iclr(review, paper, method_name, True)
    pos_reviewers = ""
    neg_reviewers = ""
    acc_reviewers = []
    rej_reviewers = []
    for reviewer in meta['summary']:
      if reviewer['rating'] > 4:
        acc_reviewers.append((reviewer['summary'], reviewer['rating']))
      else:
        rej_reviewers.append((reviewer['summary'], reviewer['rating']))
    if len(acc_reviewers) == 0 or len(rej_reviewers) == 0:
      return False
    summary_content_1 = max(acc_reviewers, key=lambda x: x[1])[0]
    summary_content_2 = min(rej_reviewers, key=lambda x: x[1])[0]
    if meta['decision'] == 'Accept':
      pos_reviewers = summary_content_1
      neg_reviewers = summary_content_2
    elif meta['decision'] == 'Reject':
      pos_reviewers = summary_content_2
      neg_reviewers = summary_content_1
    else:
      raise Exception(f"Unrecognized decision value: {meta['decision']}")

    return {
        "conversations": [
          {
            "from": "human",
            "value": f"Please provide a review based on the following paper, including a summary, strengths, weaknesses, and any questions you have: Title: {meta['title']}. Abstract: {meta['abstract']}. Main Text: {meta['paper']}."
          }
        ],
        "chosen": {
            "from": "gpt",
            "value": pos_reviewers
        },
        "rejected": {
            "from": "gpt",
            "value": neg_reviewers
        }
    }

def main():
    _, _, iclr_review_path, iclr_paper_path, _, _ = read_all_data()
    iclr_paper_path = [i for i in iclr_paper_path if not '2024' in i]
    iclr_review_path = [i for i in iclr_review_path if not '2024' in i]

    results = []
    for p, r in tqdm(zip(iclr_paper_path, iclr_review_path), desc="Conversion", total=len(iclr_paper_path)):
       result = process_dpo_data(p, r)
       results.append(result)

       
    futures = []
    inlegal_num = 0

    index = 0

    for meta in tqdm(results, desc="saving", total=len(results)):
      if meta == False:
        inlegal_num += 1
      else:
        with open(fr"./data/converted/DPO/{index:05}.json", 'w') as fp:
          json.dump(meta, fp)
          index += 1
    
    print(f"inlegal_number: {inlegal_num}/{len(results)}")
if __name__ == '__main__':
    main()
