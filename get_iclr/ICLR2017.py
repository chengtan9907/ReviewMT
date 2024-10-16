import openreview
import re
import os
import json
import time
from tqdm import tqdm
from .config import sleep_time

def get_paper_and_review(pdf_outpath, review_outpath):
    NotFoundc = 0
    client = openreview.Client(baseurl='https://api.openreview.net')
    submissions = client.get_all_notes(invitation='ICLR.cc/2017/conference/-/submission', details='directReplies')
    paper_number = 1
    c = 0
    for submission in tqdm(submissions, desc="ICLR2017_get_paper_and_review", position=1, leave=False):
        # Get paper pdf
        # if c < 212:
        #     c += 1
        #     continue
        c += 1
        if c == 213:
            continue
        continue_flag = False
        while True:
            try:
                pdf_content = client.get_pdf(submission.id)
                break
            except Exception as e:
                if "NotFoundError" in str(e):
                    NotFoundc += 1
                    print(f"{paper_number} pdf not found")
                    paper_number += 1
                    continue_flag = True
                    break
                else:
                    continue
        if continue_flag:
            continue
        with open(pdf_outpath+os.sep+f"ICLR_2017_paper_{paper_number:04d}.pdf", 'wb') as fp:
            fp.write(pdf_content)
        # pdf done
        
        meta_paper_info = {
            'id': submission.id,
            'title': submission.content['title'],
            'abstract': submission.content['abstract'],
            'reviewers': [],
            'number_of_reviewers': 0,
            'meta_review': '',
            'decision': ''
        }
        Official_Review = [reply for reply in submission.details["directReplies"] if reply["invitation"].endswith("review")]
        Official_Comment = [reply for reply in submission.details["directReplies"] if reply["invitation"].endswith("question")]
        Decision = [reply for reply in submission.details["directReplies"] if reply["invitation"].endswith("acceptance")]
        for r in Official_Review:
            meta_paper_info['number_of_reviewers'] += 1
            meta_review = {
                'id': r['id'],
                'name': r['signatures'][0].split(r'/')[-1],
                'summary': r['content']['review'],
                'confidence': '',
                'rating': r['content']['rating'],
                'rebuttal(from author)': [],
                'response(from reviewer)': []
            }
            if 'confidence' in r['content']:
                meta_review['confidence'] = r['content']['confidence'] 
            meta_paper_info['reviewers'].append(meta_review)
        for r in Official_Comment:
            if 'title' in r['content'] and r['content']['title'] == r'n/a':
                continue
            if r['signatures'][0].endswith("Authors"):# write by author
                for j in meta_paper_info['reviewers']:
                    if 'title' in r['content'] and j['name'] in r['content']['title']:
                        j['response(from reviewer)'].append(r['content']['title'] + ' ' + r['content']['comment'])
                        break
            else:# write by a reviewer
                for j in meta_paper_info['reviewers']:
                    if r['signatures'][0].endswith(j['name']):
                        if 'title' in r['content']:
                            j['rebuttal(from author)'].append(r['content']['title'] + ' ' + r['content']['question'])
                        else:
                            j['rebuttal(from author)'].append(r['content']['question'])
                        break
        for r in Decision:
                meta_paper_info['meta_review'] = r['content']['comment']
                meta_paper_info['decision'] = r['content']['decision']
        with open(review_outpath+os.sep+f"ICLR_2017_review_{paper_number:04d}.json", 'w', encoding='utf-8') as fp:
           json.dump(meta_paper_info, fp)
        paper_number += 1
        time.sleep(sleep_time)
    return len(submissions),NotFoundc