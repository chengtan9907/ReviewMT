import json
import re
import os
import shutil

def delete(string):
    pattern = r'(<.*?>)'
    l = re.findall(pattern, string)
    for i in l:
        string = string.replace(i, '')
    return string

def dealing_pdfs(paper_path, ta_path):
    dirs = os.listdir(ta_path)
    dirs.sort()
    for d in dirs:
        name = d[0:4]
        p_path = os.path.join(paper_path, f"{name} paper.md")
        t_path = os.path.join(ta_path, d)
        with open(p_path, 'r', encoding='utf-8') as fp:
            paper = fp.read()
            paper = delete(paper)
        with open(t_path, 'r', encoding='utf-8') as fp:
            ta = json.load(fp)
        title = delete(ta['title'])
        abst = delete(ta['abstract'])
        try:
            string = abst[-8:-1]
            content = paper.split(string)[1:]
        except:
            print(name)
        if len(content) != 1:
            content = " ".join(content)
        ta['content'] = content
        with open(t_path, 'w', encoding='utf-8') as fp:
            json.dump(ta, fp)

def extract_review(review_path):
    with open(review_path,'r', encoding='utf-8') as fp:
        review = fp.read()
    pattern = r"Reviewer #(\d+)\s*(.*?)\n\n(.*?)\n\n"
    matches = re.findall(pattern, review, re.DOTALL | re.MULTILINE)
    reviewers_info = []
    catch_flag = False
    if not len(matches)==0:
        for match in matches:
            reviewer_number, reviewer_name, review_content = match
            meta = {
                'reviewer_name': f"Reviewer {reviewer_number}",
                'review_content': review_content.strip()
            }
            reviewers_info.append(meta)
        catch_flag = True
    review_pattern = re.compile(r"Reviewer #(\d+)\s*\(Remarks to the Author\):(.*?)(?=Reviewer #\d+|\Z)", re.DOTALL)
    matches = review_pattern.findall(review)
    if not len(matches)==0:
        for i, (reviewer_number, review_content) in enumerate(matches):
            meta = {
                'reviewer_name': f"Reviewer {reviewer_number}",
                'review_content': review_content.strip()
            }
            reviewers_info.append(meta)
        catch_flag = True

    reviewer_pattern = re.compile(r'Reviewer #(\d+):')
    comment_pattern = re.compile(r'Reviewer #\d+:.*?(?=Reviewer #\d+:|$)', re.DOTALL)
    reviewer_matches = reviewer_pattern.finditer(review)
    reviewers_comments = dict()
    for reviewer_match in reviewer_matches:
        reviewer_number = reviewer_match.group(1)
        comment_match = comment_pattern.search(review, reviewer_match.end())
        if comment_match:
            reviewers_comments[reviewer_number] = comment_match.group(0).strip()
    if not len(reviewers_comments) == 0:
        for i,j in reviewers_comments.items():
            if re.search("RESPONSE TO REFEREES", j, re.IGNORECASE):
                j = re.split("RESPONSE TO REFEREES", j, re.IGNORECASE)[0]
            if re.search(r"^Reviewer #\d: ", j):
                i = re.search(r"^Reviewer #(\d): ", j).group(1)
                j = re.split(r"^Reviewer #\d: ", j)[1]
                meta = {
                    'reviewer_name': f"Reviewer {i}",
                    'review_content': j
                }
                reviewers_info.append(meta)
                catch_flag = True
    return reviewers_info

def make_nature(title, abst, content, review, decision):
    review_text = ""
    for r in review:
        review_text += f"{r['reviewer_name']} {r['review_content']}\n"
    meta = {
        "instruction": "Role: Decision Maker. Task: Review all responses from the author and comments from all reviewers to provide the meta-review and determine the final decision. Explicitly state 'Accept' or 'Reject' at the end of your output.",
        "input": f"Title: {title} Abstract: {abst} Main Text: {content}",
        "output": f"{review_text} Final decision: {decision}",
        "system": "",
        "history": ""
    }
    return meta


def clean(outpath):
    pdirs = os.listdir(outpath)
    pdirs.sort()
    for pd in pdirs:
        outpath2 = os.path.join(outpath,pd)
        dirs = os.listdir(outpath2)
        dirs.sort()
        for d in dirs:
            outpath3 = os.path.join(outpath2, d)
            if d.endswith(".md"):
                os.rename(outpath3, os.path.join(outpath, d))
        shutil.rmtree(os.path.join(outpath2))

def make_iclr(papermd_path, review_path):
    with open(papermd_path, 'r') as fp1, open(review_path, 'r') as fp2:
        paper = fp1.read()
        review = json.load(fp2)
    title = review['title']
    abst = review['abstract']
    if len(paper.split(r"## 1 Introduction")) == 2: 
        content = r"## 1 Introduction" + paper.split(r"## 1 Introduction")[1]
    elif len(paper.split(r"## 1 Introdution")) == 2:
        content = r"## 1 Introduction" + paper.split(r"## 1 Introdution")[1]
    else:
        raise Exception("Unknow paper format")
    meta = {
        "instruction": "Role: Decision Maker. Task: Review all responses from the author and comments from all reviewers to provide the meta-review and determine the final decision. Explicitly state 'Accept' or 'Reject' at the end of your output.",
        "input": f"Title: {title} Abstract: {abst} Main Text: {content}",
        "output": f"{review['meta_review']} Final decision: {review['decision']}",
        "system": "",
        "history": []
    }
    review_number = 1
    for r in review['reviewers']:
        score = re.findall(r'^\d+', r['rating'])
        if not len(score) == 1:
            raise Exception("Error！")
        score = int(score[0])
        ST = "Tolerent" if score > 5 else "Strict"
        S_W = ""
        if 'strengths' in r:
            S_W = f" Strengths and Weaknesses: {r['strengths']}\n{r['weakness']}"
        elif 'strengths_and_weakness' in r:
            S_W = f" Strengths and Weaknesses: {r['strengths_and_weakness']}"
        else:
            S_W = ""
        summary = r['summary']
        confidence = r['confidence']
        if 'question' in r:
            question = f" Questions: {r['questions']}"
        else:
            question = ""
        if len(r['rebuttal(from author)']) != 0:
            rebuttal = ""
            for c in r['rebuttal(from author)']:
                rebuttal = rebuttal + c + '\n'
        else:
            rebuttal = "Author does't have any rebuttal."
        if len(r['response(from reviewer)']) != 0:
            response = ""
            for c in r['response(from reviewer)']:
                response = response + c + '\n'
        else:
            response = f"Reviewer {review_number} does't have any response to the Author."
        meta1 = [f"Role: Reviewer {review_number}. Style: {ST}. Task: Provide a critical review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have. Please provide your review of '{title}'", f"Summary: {summary}{S_W}{question}"]
        meta2 = [f"Role: Author. Task: Respond to Reviewer {review_number}'s comments by clarifying the mentioned weaknesses and answering the posed questions. Summarize the Reviewer {review_number}'s comment above and provide a response accordingly.", f"{rebuttal}"]
        meta3 = [f"Role: Reviewer {review_number}. Style: {ST}. Task: Based on the author's response, provide a final score from 1 to 10 and a confidence from 1 to 5.", f"{response} Score: {r['rating']} Confidence: {confidence}"]
        meta['history'].append(meta1)
        meta['history'].append(meta2)
        meta['history'].append(meta3)
        review_number += 1
    return meta