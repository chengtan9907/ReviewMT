import argparse
import os

META_DATA_TEMPLATE={
    'title': '',
    'abstract': '',
    'paper': '',
    'meta_review': '',
    'decision': '',
    'history': []
}

def convert(metaData):
    result = dict()
    result['instruction'] = "You are a decision maker. Please review all responses from the author and comments from all reviewers to provide the meta-review and determine the final decision. Explicitly state 'Accept' or 'Reject' at the end of your output."
    result['input'] = f"Title: {metaData['title']}. Abstract: {metaData['abstract']}. Main Text: {metaData['paper']}"
    result['output'] = f"{metaData['meta_review']}. Final decision: {metaData['decision']}."
    result['system'] = ""
    result['history'] = metaData['history']
    return result

class ICLR_Formatter:
    meta = None
    def base(self, review, paper):
        self.meta = None
        self.meta = META_DATA_TEMPLATE.copy()
        self.meta['title'] = review['title']
        self.meta['abstract'] = review['abstract']
        self.meta['paper'] = paper
        self.meta['meta_review'] = review['meta_review']
        self.meta['decision'] = review['decision'].split('(')[0].strip()
    def deal2017(self, reviewers):
        meta = self.meta
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
                response = f"Reviewer{str(index+1)} doesn't have more comment."
            
            meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Please provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
            f"Summary: {reviewer['summary']}."])
            meta['history'].append([f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"])
            meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
            f"{response} Score: {rating}. Confidence: {reviewer['confidence']}."])
        return convert(meta)
        
    def deal2018(self, reviewers):
        meta = self.meta
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
                response = f"Reviewer{str(index+1)} doesn't have more comment."
            
            meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Please provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
            f"Summary: {reviewer['summary']}."])
            meta['history'].append([f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"])
            meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
            f"{response} Score: {rating}. Confidence: {reviewer['confidence']}."])
        return convert(meta)
        
    def deal2019(self, reviewers):
        meta = self.meta
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
                response = f"Reviewer{str(index+1)} doesn't have more comment."
            
            meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Please provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
            f"Summary: {reviewer['summary']}."])
            meta['history'].append([f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"])
            meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
            f"{response} Score: {rating}. Confidence: {reviewer['confidence']}."])
        return convert(meta)
        
    def deal2020(self, reviewers):
        meta = self.meta
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
                response = f"Reviewer{str(index+1)} doesn't have more comment."
            
            meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Please provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
            f"Summary: {reviewer['summary']}."])
            meta['history'].append([f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"])
            meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
            f"{response} Score: {rating}."])
        return convert(meta)
        
    def deal2021(self, reviewers):
        meta = self.meta
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
                response = f"Reviewer{str(index+1)} doesn't have more comment."
            
            meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Pleasse provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
            f"Summary: {reviewer['summary']}."])
            meta['history'].append([f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"])
            meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
            f"{response} Score: {rating}. Confidence: {reviewer['confidence']}."])
        return convert(meta)
        
    def deal2022(self, reviewers):
        meta = self.meta
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
                response = f"Reviewer{str(index+1)} doesn't have more comment."
            
            meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Please provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
            f"Summary: {reviewer['summary']}. Strengths and Weaknesses: {reviewer['strengths_and_weakness']}."])
            meta['history'].append([f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"])
            meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
            f"{response} Score: {rating}. Confidence: {reviewer['confidence']}."])
        return convert(meta)
        
    def deal2023(self, reviewers):
        meta = self.meta
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
                response = f"Reviewer{str(index+1)} doesn't have more comment."
            
            meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Please provide a review based on the paper provided, including a summary, strengths, weaknesses, and any questions you have.",
            f"Summary: {reviewer['summary']}. Strengths and Weaknesses: {reviewer['strengths_and_weakness']}."])
            meta['history'].append([f"You are the authors. Please respond to Reviewer {str(index+1)}'s comments by clarifying the mentioned weaknesses and answering the posed questions.", f"{rebuttal}"])
            meta['history'].append([f"You are Reviewer {str(index+1)}, and your review style is {style}. Based on the author's response, please provide a final score from 1 to 10 and a confidence from 1 to 5.",
            f"{response} Score: {rating}. Confidence: {reviewer['confidence']}."])
        return convert(meta)

    def deal2024(self, reviewers):
        meta = self.meta
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
                response = f"Reviewer{str(index+1)} doesn't have more comment."
            
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
            response = f"Reviewer{str(index+1)} doesn't have more comment."
        
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
    meta['title'] = review['title']
    meta['abstract'] = review['abstract']
    meta['paper'] = paper
    meta['meta_review'] = review['meta_review']
    meta['decision'] = review['decision'].split('(')[0].strip()
    for index, reviewer in enumerate(review['reviewers']):
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
            response = f"Reviewer{str(index+1)} doesn't have more comment."
        
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