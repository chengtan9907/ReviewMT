import argparse
import os
import re

META_DATA_TEMPLATE={
    'title': '',
    'abstract': '',
    'paper': '',
    'meta_review': '',
    'decision': '',
    'history': []
}

def extract_single_pdf(single_md_path):
    with open(single_md_path, 'r') as fp:
        paper = fp.read()
    content = re.split(r'(?i)(introd)', paper, maxsplit=1)[1:]
    content = "".join(content)
    return content

def convert(metaData):
    result = dict()
    result['instruction'] = "You are a decision maker. Please review all responses from the author and comments from all reviewers to provide the meta-review and determine the final decision. Explicitly state 'Accept' or 'Reject' at the end of your output."
    result['input'] = f"Title: {metaData['title']}. Abstract: {metaData['abstract']}. Main Text: {metaData['paper']}"
    result['output'] = f"{metaData['meta_review']}. Final decision: {metaData['decision']}."
    result['system'] = ""
    result['history'] = metaData['history']
    return result


def process_iclr(review, paper, method_name, DPO_type=False):
    meta = None
    meta = None
    meta = META_DATA_TEMPLATE.copy()
    meta['title'] = review['title']
    meta['abstract'] = review['abstract']
    meta['paper'] = paper
    meta['meta_review'] = review['meta_review']
    meta['decision'] = review['decision'].split('(')[0].strip()
    if "Accept" in meta['decision']:
        meta['decision'] = "Accept"
    elif meta['decision'] == "Invite to Workshop Track":
        meta['decision'] = "Reject"
    if '2017' in method_name:
        result = deal2017(review['reviewers'], meta, DPO_type)
    elif '2018' in method_name:
        result = deal2018(review['reviewers'], meta, DPO_type)
    elif '2019' in method_name:
        result = deal2019(review['reviewers'], meta, DPO_type)
    elif '2020' in method_name:
        result = deal2020(review['reviewers'], meta, DPO_type)
    elif '2021' in method_name:
        result = deal2021(review['reviewers'], meta, DPO_type)
    elif '2022' in method_name:
        result = deal2022(review['reviewers'], meta, DPO_type)
    elif '2023' in method_name:
        result = deal2023(review['reviewers'], meta, DPO_type)
    elif '2024' in method_name:
        result = deal2024(review['reviewers'], meta, DPO_type)
    return result

def deal2017(reviewers, meta, DPO_type):
    dpo_meta = []
    meta['history'] = []
    for index, reviewer in enumerate(reviewers):
        rating = int(reviewer['rating'][0])
        if rating >= 1 and rating <= 3:
            style = 'Harsh'
        elif rating >=4 and rating <= 6:
            style = 'Neutral'
        elif rating >=7 and rating <= 10:
            style = 'Mild'
        else:
            raise Exception("Rating value error!")
        
        rebuttal = " ".join(reviewer['rebuttal(from author)']).strip()
        if not rebuttal:
            rebuttal = "The author doesn't have any rebuttal."
        response = " ".join(reviewer['response(from reviewer)']).strip()
        if not response:
            response = f"Reviewer {str(index+1)} doesn't have more comment."
        
        meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Please provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
        f"Summary: {reviewer['summary']}."])
        meta['history'].append([f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"])
        meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
        f"{response} Score: {rating}. Confidence: {reviewer['confidence']}."])
        dpo_meta.append({
            "summary": f"Summary: {reviewer['summary']}.",
            "rating": rating
        })
    if not DPO_type:
        return convert(meta)
    else:
        return {
            'title': meta['title'],
            'abstract': meta['abstract'],
            'paper': meta['paper'],
            'decision': meta['decision'],
            'summary': dpo_meta
        }
    
def deal2018(reviewers, meta, DPO_type):
    dpo_meta = []
    meta['history'] = []
    for index, reviewer in enumerate(reviewers):
        rating = int(reviewer['rating'][0])
        if rating >= 1 and rating <= 3:
            style = 'Harsh'
        elif rating >=4 and rating <= 6:
            style = 'Neutral'
        elif rating >=7 and rating <= 10:
            style = 'Mild'
        else:
            raise Exception("Rating value error!")
        
        rebuttal = " ".join(reviewer['rebuttal(from author)']).strip()
        if not rebuttal:
            rebuttal = "The author doesn't have any rebuttal."
        response = " ".join(reviewer['response(from reviewer)']).strip()
        if not response:
            response = f"Reviewer {str(index+1)} doesn't have more comment."

        meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Please provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
        f"Summary: {reviewer['summary']}."])
        meta['history'].append([f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"])
        meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
        f"{response} Score: {rating}. Confidence: {reviewer['confidence']}."])
        dpo_meta.append({
            "summary": f"Summary: {reviewer['summary']}.",
            "rating": rating
        })
    if not DPO_type:
        return convert(meta)
    else:
        return {
            'title': meta['title'],
            'abstract': meta['abstract'],
            'paper': meta['paper'],
            'decision': meta['decision'],
            'summary': dpo_meta
        }
    
def deal2019(reviewers, meta, DPO_type):
    dpo_meta = []
    meta['history'] = []
    for index, reviewer in enumerate(reviewers):
        rating = int(reviewer['rating'][0])
        if rating >= 1 and rating <= 3:
            style = 'Harsh'
        elif rating >=4 and rating <= 6:
            style = 'Neutral'
        elif rating >=7 and rating <= 10:
            style = 'Mild'
        else:
            raise Exception("Rating value error!")
        
        rebuttal = " ".join(reviewer['rebuttal(from author)']).strip()
        if not rebuttal:
            rebuttal = "The author doesn't have any rebuttal."
        response = " ".join(reviewer['response(from reviewer)']).strip()
        if not response:
            response = f"Reviewer {str(index+1)} doesn't have more comment."
        
        meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Please provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
        f"Summary: {reviewer['summary']}."])
        meta['history'].append([f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"])
        meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
        f"{response} Score: {rating}. Confidence: {reviewer['confidence']}."])
        dpo_meta.append({
            "summary": f"Summary: {reviewer['summary']}.",
            "rating": rating
        })
    if not DPO_type:
        return convert(meta)
    else:
        return {
            'title': meta['title'],
            'abstract': meta['abstract'],
            'paper': meta['paper'],
            'decision': meta['decision'],
            'summary': dpo_meta
        }
    
def deal2020(reviewers, meta, DPO_type):
    dpo_meta = []
    meta['history'] = []
    for index, reviewer in enumerate(reviewers):
        rating = int(reviewer['rating'][0])
        if rating >= 1 and rating <= 3:
            style = 'Harsh'
        elif rating >=4 and rating <= 6:
            style = 'Neutral'
        elif rating >=7 and rating <= 10:
            style = 'Mild'
        else:
            raise Exception("Rating value error!")
        
        rebuttal = " ".join(reviewer['rebuttal(from author)']).strip()
        if not rebuttal:
            rebuttal = "The author doesn't have any rebuttal."
        response = " ".join(reviewer['response(from reviewer)']).strip()
        if not response:
            response = f"Reviewer {str(index+1)} doesn't have more comment."
        
        meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Please provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
        f"Summary: {reviewer['summary']}."])
        meta['history'].append([f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"])
        meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
        f"{response} Score: {rating}."])
        dpo_meta.append({
            "summary": f"Summary: {reviewer['summary']}.",
            "rating": rating
        })
    if not DPO_type:
        return convert(meta)
    else:
        return {
            'title': meta['title'],
            'abstract': meta['abstract'],
            'paper': meta['paper'],
            'decision': meta['decision'],
            'summary': dpo_meta
        }
    
def deal2021(reviewers, meta, DPO_type):
    dpo_meta = []
    meta['history'] = []
    for index, reviewer in enumerate(reviewers):
        rating = int(reviewer['rating'][0])
        if rating >= 1 and rating <= 3:
            style = 'Harsh'
        elif rating >=4 and rating <= 6:
            style = 'Neutral'
        elif rating >=7 and rating <= 10:
            style = 'Mild'
        else:
            raise Exception("Rating value error!")
        
        rebuttal = " ".join(reviewer['rebuttal(from author)']).strip()
        if not rebuttal:
            rebuttal = "The author doesn't have any rebuttal."
        response = " ".join(reviewer['response(from reviewer)']).strip()
        if not response:
            response = f"Reviewer {str(index+1)} doesn't have more comment."
        
        meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Pleasse provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
        f"Summary: {reviewer['summary']}."])
        meta['history'].append([f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"])
        meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
        f"{response} Score: {rating}. Confidence: {reviewer['confidence']}."])
        dpo_meta.append({
            "summary": f"Summary: {reviewer['summary']}.",
            "rating": rating
        })
    if not DPO_type:
        return convert(meta)
    else:
        return {
            'title': meta['title'],
            'abstract': meta['abstract'],
            'paper': meta['paper'],
            'decision': meta['decision'],
            'summary': dpo_meta
        }
    
def deal2022(reviewers, meta, DPO_type):
    dpo_meta = []
    meta['history'] = []
    for index, reviewer in enumerate(reviewers):
        rating = int(reviewer['rating'][0])
        if rating >= 1 and rating <= 3:
            style = 'Harsh'
        elif rating >=4 and rating <= 6:
            style = 'Neutral'
        elif rating >=7 and rating <= 10:
            style = 'Mild'
        else:
            raise Exception("Rating value error!")
        
        rebuttal = " ".join(reviewer['rebuttal(from author)']).strip()
        if not rebuttal:
            rebuttal = "The author doesn't have any rebuttal."
        response = " ".join(reviewer['response(from reviewer)']).strip()
        if not response:
            response = f"Reviewer {str(index+1)} doesn't have more comment."
        
        meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Please provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
        f"Summary: {reviewer['summary']}. Strengths and Weaknesses: {reviewer['strengths_and_weakness']}."])
        meta['history'].append([f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"])
        meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
        f"{response} Score: {rating}. Confidence: {reviewer['confidence']}."])
        dpo_meta.append({
            "summary": f"Summary: {reviewer['summary']}. Strengths and Weaknesses: {reviewer['strengths_and_weakness']}.",
            "rating": rating
        })
    if not DPO_type:
        return convert(meta)
    else:
        return {
            'title': meta['title'],
            'abstract': meta['abstract'],
            'paper': meta['paper'],
            'decision': meta['decision'],
            'summary': dpo_meta
        }
    
def deal2023(reviewers, meta, DPO_type):
    dpo_meta = []
    meta['history'] = []
    for index, reviewer in enumerate(reviewers):
        rating = int(reviewer['rating'][0])
        if rating >= 1 and rating <= 3:
            style = 'Harsh'
        elif rating >=4 and rating <= 6:
            style = 'Neutral'
        elif rating >=7 and rating <= 10:
            style = 'Mild'
        else:
            raise Exception("Rating value error!")
        
        rebuttal = " ".join(reviewer['rebuttal(from author)']).strip()
        if not rebuttal:
            rebuttal = "The author doesn't have any rebuttal."
        response = " ".join(reviewer['response(from reviewer)']).strip()
        if not response:
            response = f"Reviewer {str(index+1)} doesn't have more comment."
        
        meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Please provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
        f"Summary: {reviewer['summary']}. Strengths and Weaknesses: {reviewer['strengths_and_weakness']}."])
        meta['history'].append([f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"])
        meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
        f"{response} Score: {rating}. Confidence: {reviewer['confidence']}."])
        dpo_meta.append({
            "summary": f"Summary: {reviewer['summary']}. Strengths and Weaknesses: {reviewer['strengths_and_weakness']}.",
            "rating": rating
        })
    if not DPO_type:
        return convert(meta)
    else:
        return {
            'title': meta['title'],
            'abstract': meta['abstract'],
            'paper': meta['paper'],
            'decision': meta['decision'],
            'summary': dpo_meta
        }

def deal2024(reviewers, meta, DPO_type):
    meta['history'] = []
    for index, reviewer in enumerate(reviewers):
        rating = int(reviewer['rating'][0])
        if rating >= 1 and rating <= 3:
            style = 'Harsh'
        elif rating >=4 and rating <= 6:
            style = 'Neutral'
        elif rating >=7 and rating <= 10:
            style = 'Mild'
        else:
            raise Exception("Rating value error!")
        
        rebuttal = " ".join(reviewer['rebuttal(from author)']).strip()
        if not rebuttal:
            rebuttal = "The author doesn't have any rebuttal."
        response = " ".join(reviewer['response(from reviewer)']).strip()
        if not response:
            response = f"Reviewer {str(index+1)} doesn't have more comment."
        
        meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Please provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
        f"Summary: {reviewer['summary']}. Strengths and Weaknesses: {reviewer['strengths'] + '. ' + reviewer['weakness']}. Questions: {reviewer['questions']}."])
        meta['history'].append([f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"])
        meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
        f"{response} Score: {rating}. Confidence: {reviewer['confidence']}."])
    return convert(meta)

def uai(review, paper):
    meta = META_DATA_TEMPLATE.copy()
    meta['title'] = review['title']
    meta['abstract'] = review['abstract']
    meta['paper'] = paper
    meta['meta_review'] = review['meta_review']
    meta['decision'] = review['decision'].split('(')[0].strip()
    meta['history'] = []
    for index, reviewer in enumerate(review['reviewers']):
        rating = int(reviewer['summary']['Q6 Overall score'][0]) 
        if rating >= 1 and rating <= 3:
            style = 'Harsh'
        elif rating >=4 and rating <= 6:
            style = 'Neutral'
        elif rating >=7 and rating <= 10:
            style = 'Mild'
        else:
            raise Exception("Rating value error!")
        
        rebuttal = " ".join(reviewer['rebuttal(from author)']).strip()
        if not rebuttal:
            rebuttal = "The author doesn't have any rebuttal."
        response = " ".join(reviewer['response(from reviewer)']).strip()
        if not response:
            response = f"Reviewer {str(index+1)} doesn't have more comment."
        
        summarylist = [reviewer['summary']['Q1 Summary and contributions'], reviewer['summary']['Q2 Assessment of the paper'], reviewer['summary']['Q5 Detailed comments to the authors'], reviewer['summary']['Q7 Justification for your score']]
        summary = " ".join(summarylist)
        
        meta['history'].append(
            [f"You are Reviewer {str(index+1)}, and your review style is {style}. Please provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
        f"{summary}."]
        )
        meta['history'].append(
            [f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"]
        )
        meta['history'].append(
            [f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
       f"{response} Score: {rating}. Confidence: {reviewer['summary']['Q8 Confidence in your score']}."]
        )
    return convert(meta)

def nips(review, paper):
    meta = META_DATA_TEMPLATE.copy()
    if isinstance(review['title'], dict):
        meta['title'] = review['title']['value']
    else:
        meta['title'] = review['title']
    if isinstance(review['abstract'], dict):
        meta['abstract'] = review['abstract']['value']
    else:
        meta['abstract'] = review['abstract']
    if isinstance(review['meta_review'], dict):
        meta['meta_review'] = review['meta_review']['value']
    else:
        meta['meta_review'] = review['meta_review']
    if isinstance(review['decision'], dict):
        meta['decision'] = review['decision']['value'].split('(')[0].strip()
    else:
        meta['decision'] = review['decision'].split('(')[0].strip()
    meta['paper'] = paper
    meta['history'] = []
    for index, reviewer in enumerate(review['reviewers']):
        if isinstance(reviewer['rating'], dict):
            rating = int(reviewer['rating']['value'][0]) 
        else:
            rating = int(reviewer['rating'][0]) 

        if rating >= 1 and rating <= 3:
            style = 'Harsh'
        elif rating >=4 and rating <= 6:
            style = 'Neutral'
        elif rating >=7 and rating <= 10:
            style = 'Mild'
        else:
            raise Exception("Rating value error!")
        
        if (len(reviewer['rebuttal(from author)']) != 0) and (isinstance(reviewer['rebuttal(from author)'][0], dict)):
            rebuttal = ""
            for i in reviewer['rebuttal(from author)']:
                rebuttal += (i['value']).strip()
        else:
            rebuttal = " ".join(reviewer['rebuttal(from author)']).strip()
        if not rebuttal:
            rebuttal = "The author doesn't have any rebuttal."
        
        if (len(reviewer['response(from reviewer)']) != 0) and (isinstance(reviewer['response(from reviewer)'][0], dict)):
            response = ""
            for i in reviewer['response(from reviewer)']:
                response += (i['value']).strip()
        else:
            response = " ".join(reviewer['response(from reviewer)']).strip()
        if not response:
            response = f"Reviewer {str(index+1)} doesn't have more comment."
        
        meta['history'].append(
            [f"You are Reviewer {str(index+1)}, and your review style is {style}. Please provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
        f"{reviewer['summary']}."]
        )
        meta['history'].append(
            [f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"]
        )
        meta['history'].append(
            [f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
       f"{response} Score: {rating}. Confidence: {reviewer['confidence']}."]
        )
    return convert(meta)