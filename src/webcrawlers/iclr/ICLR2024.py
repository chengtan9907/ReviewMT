import openreview
import re
import os
import json
from tqdm import tqdm

def get_paper_and_review(pdf_outpath, review_outpath):
    NotFoundc = 0
    client = openreview.Client(baseurl='https://api2.openreview.net')
    submissions = client.get_all_notes(invitation='ICLR.cc/2024/Conference/-/Submission', details='directReplies')
    paper_number = 1
    for submission in tqdm(submissions, desc="ICLR2024_get_paper_and_review", position=1, leave=False):
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
        if(os.getcwd().endswith('Final Version')):
            os.chdir("webcrawlers")
        with open(pdf_outpath+os.sep+f"ICLR_2024_paper_{paper_number:04d}.pdf", 'wb') as fp:
            fp.write(pdf_content)
        # pdf done
        
        meta_paper_info = {
            'id': submission.id,
            'title': submission.content['title']['value'],
            'abstract': submission.content['abstract']['value'],
            'reviewers': [],
            'number_of_reviewers': 0,
            'meta_review': '',
            'decision': ''
        }
        # reviewers, meta_review, and decision contents are not saved for retracted type papers
        if submission.content['venue']['value'] == "ICLR 2024 Conference Withdrawn Submission":
            meta_paper_info['decision'] = 'Withdrawn'
        else:
            Official_Review = [reply for reply in submission.details["directReplies"] if reply["invitations"][0].endswith("Official_Review")]
            Official_Comment = [reply for reply in submission.details["directReplies"] if reply["invitations"][0].endswith("Official_Comment")]
            Meta_Review = [reply for reply in submission.details["directReplies"] if reply["invitations"][0].endswith("Meta_Review")]
            Decision = [reply for reply in submission.details["directReplies"] if reply["invitations"][0].endswith("Decision")]
            for r in Official_Review:
                meta_paper_info['number_of_reviewers'] += 1
                meta_review = {
                    'id': r['id'],
                    'name': r['signatures'][0].split(r'/')[-1].split(r'_')[-1],
                    'summary': r['content']['summary']['value'],
                    'confidence': r['content']['confidence']['value'],
                    'strengths': r['content']['strengths']['value'],
                    'weakness': r['content']['weaknesses']['value'],
                    'rating': r['content']['rating']['value'],
                    'questions': r['content']['questions']['value'],
                    'rebuttal(from author)': [],
                    'response(from reviewer)': []
                }
                meta_paper_info['reviewers'].append(meta_review)
            for r in Official_Comment:
                    if r['signatures'][0].endswith("Authors"):# write by author
                        for j in meta_paper_info['reviewers']:
                            if 'title' in r['content'] and j['name'] in r['content']['title']['value']:
                                j['response(from reviewer)'].append(r['content']['title']['value'] + ' ' + r['content']['comment']['value'])
                                break
                    else:# write by a reviewer
                        for j in meta_paper_info['reviewers']:
                            if r['signatures'][0].endswith(j['name']):
                                if 'title' in r['content']:
                                    j['rebuttal(from author)'].append(r['content']['title']['value'] + ' ' + r['content']['comment']['value'])
                                else:
                                    j['rebuttal(from author)'].append(r['content']['comment']['value'])
                                break
            for r in Meta_Review:
                    meta_paper_info['meta_review'] = r['content']['metareview']['value']
            for r in Decision:
                    meta_paper_info['decision'] = r['content']['decision']['value']
        with open(review_outpath+os.sep+f"ICLR_2024_review_{paper_number:04d}.json", 'w', encoding='utf-8') as fp:
           json.dump(meta_paper_info, fp)
        paper_number += 1
    return len(submissions),NotFoundc