import openreview
import re
import os
import json
from tqdm import tqdm

def get_paper_and_review(pdf_outpath, review_outpath):
    NotFoundc = 0
    client = openreview.Client(baseurl='https://api.openreview.net')
    submissions = client.get_all_notes(invitation='ICLR.cc/2020/Conference/-/Blind_Submission', details='directReplies')
    paper_number = 1
    for submission in tqdm(submissions, desc="ICLR2020_get_paper_and_review", position=1, leave=False):
        # Get paper pdf
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
        with open(pdf_outpath+os.sep+f"ICLR_2020_paper_{paper_number:04d}.pdf", 'wb') as fp:
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
        Official_Review = [reply for reply in submission.details["directReplies"] if reply["invitation"].endswith("Official_Review")]
        Official_Comment = [reply for reply in submission.details["directReplies"] if reply["invitation"].endswith("Official_Comment")]
        Decision = [reply for reply in submission.details["directReplies"] if reply["invitation"].endswith("Decision")]
        for r in Official_Review:
            meta_paper_info['number_of_reviewers'] += 1
            meta_review = {
                'id': r['id'],
                'name': r['signatures'][0].split(r'/')[-1],
                'summary': r['content']['review'],
                'rating': r['content']['rating'],
                'rebuttal(from author)': [],
                'response(from reviewer)': []
            }
            meta_paper_info['reviewers'].append(meta_review)
        for r in Official_Comment:
                    if r['signatures'][0].endswith("Authors"):# write by author
                        for j in meta_paper_info['reviewers']:
                            if 'title' in r['content'] and j['name'] in r['content']['title']:
                                j['response(from reviewer)'].append(r['content']['title'] + ' ' + r['content']['comment'])
                                break
                    else:# write by a reviewer
                        for j in meta_paper_info['reviewers']:
                            if r['signatures'][0].endswith(j['name']):
                                if 'title' in r['content']:
                                    j['rebuttal(from author)'].append(r['content']['title'] + ' ' + r['content']['comment'])
                                else:
                                    j['rebuttal(from author)'].append(r['content']['comment'])
                                break
        for r in Decision:
                meta_paper_info['meta_review'] = r['content']['comment']
                meta_paper_info['decision'] = r['content']['decision']
        with open(review_outpath+os.sep+f"ICLR_2020_review_{paper_number:04d}.json", 'w', encoding='utf-8') as fp:
           json.dump(meta_paper_info, fp)
        paper_number += 1
    return len(submissions),NotFoundc