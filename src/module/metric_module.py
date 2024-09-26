import os
import json
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import re
from sklearn.metrics import f1_score
from bert_score import score as bs
from tqdm import tqdm

os.environ['NLTK_DATA'] = './cache_download'

# Initialize scorers
bleu_smoothie = SmoothingFunction().method4
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calculate_bleu(reference, hypothesis):
    bleu4 = sentence_bleu([reference], hypothesis, smoothing_function=bleu_smoothie)
    bleu2 = sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=bleu_smoothie)
    return bleu4, bleu2

def calculate_rouge(reference, hypothesis):
    scores = rouge_scorer.score(reference, hypothesis)
    rouge_l = scores['rougeL'].fmeasure
    rouge_1 = scores['rouge1'].fmeasure
    rouge_2 = scores['rouge2'].fmeasure
    return rouge_l, rouge_1, rouge_2

def calculate_meteor(reference, hypothesis):
    try:
        score = meteor_score([reference], hypothesis)
    except:
        os.environ["http_proxy"] = "http://wolfcave.myds.me:987"
        os.environ["https_proxy"] = "http://wolfcave.myds.me:987"
        os.environ["all_proxy"] = "http://wolfcave.myds.me:988"
        
        nltk.download('wordnet')
        score = meteor_score([reference], hypothesis)
    return score

def calculate_bert(reference, hypothesis):
    _, _, f1 = bs(hypothesis, reference, lang='en', model_type='bert-base-uncased')
    return f1
 
def process_files(path, dataset='iclr', total=100):    
    bleu2_list, bleu4_list, rouge_l_list, rouge1_list, rouge2_list, meteor_list = [], [], [], [], [], []
    bert_list = []
    mae_list = []
    gt_decision_lst, pr_decision_lst = [], []
    missed = 0
    missed_reviews, total_reviews = 0, 0
    missed_decisions, total_decisions = 0, 0

    for file_path in tqdm(path):
        if not os.path.isfile(file_path):
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as fp:
                pred_data = json.load(fp)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error reading {file_path}: {e}")
            continue

        roles = pred_data.get('roles', [])
        gts = pred_data.get('gt_replies', [])
        prs = pred_data.get('pred_replies', [])

        if len(gts) == 0 or len(prs) == 0:
            missed += 1
            continue

        for idx, (role, gt_reply, pr_reply) in enumerate(zip(roles, gts, prs)):
            gt_reply, pr_reply = gt_reply.lower(), pr_reply.lower()

            # text similarity
            bert = calculate_bert([gt_reply], [pr_reply])
            bleu4, bleu2 = calculate_bleu(gt_reply.split(), pr_reply.split())
            rouge_l, rouge1, rouge2 = calculate_rouge(gt_reply, pr_reply)
            meteor = calculate_meteor(gt_reply.split(), pr_reply.split())
            

            # reviewer score
            # if role == 'reviewer' and (idx + 1) % 3 == 0:
            print(role)
            if role == 'reviewer':
                total_reviews += 1

                gt_score_pattern = r"score: (\d+)"
                try:
                    gt_score = int(re.findall(gt_score_pattern, gt_reply)[0])
                except:
                    continue
                try:
                    pr_score_pattern = r"score: .{0,10}" # the first number after "score" within the 10 characters
                    pr_score = int(re.findall(r"\d+", re.findall(pr_score_pattern, pr_reply)[0])[0])
                    assert pr_score <= 10 and pr_score >= 1
                    mae_list.append(np.abs(gt_score-pr_score))
                except:
                    missed_reviews += 1

            # decision score
            if role == 'decision maker':
                total_decisions += 1

                gt_decision_pattern = r"final decision: (\w+)"
                gt_decision = re.search(gt_decision_pattern, gt_reply).group(1)
                gt_decision = 1 if gt_decision == 'accept' else 0
                try:
                    accept_matches = len(re.findall(r"accept", pr_reply, re.IGNORECASE))
                    reject_matches = len(re.findall(r"reject", pr_reply, re.IGNORECASE))
                    if accept_matches == reject_matches:
                        missed_decisions += 1
                    elif accept_matches > reject_matches:
                        pr_decision = 1
                        gt_decision_lst.append(gt_decision)
                        pr_decision_lst.append(pr_decision)
                    else:
                        pr_decision = 0
                        gt_decision_lst.append(gt_decision)
                        pr_decision_lst.append(pr_decision)
                except:
                    missed_decisions += 1

            bert_list.append(bert)
            bleu2_list.append(bleu2)
            bleu4_list.append(bleu4)
            rouge_l_list.append(rouge_l)
            rouge1_list.append(rouge1)
            rouge2_list.append(rouge2)
            meteor_list.append(meteor)

    return {
        'bleu2_mean': np.mean(bleu2_list),
        'bleu2_std': np.std(bleu2_list),
        'bleu4_mean': np.mean(bleu4_list),
        'bleu4_std': np.std(bleu4_list),
        'bert_mean': np.mean(bert_list),
        'bert_std': np.std(bert_list),
        'rouge_l_mean': np.mean(rouge_l_list),
        'rouge_l_std': np.std(rouge_l_list),
        'rouge1_mean': np.mean(rouge1_list),
        'rouge1_std': np.std(rouge1_list),
        'rouge2_mean': np.mean(rouge2_list),
        'rouge2_std': np.std(rouge2_list),
        'meteor_mean': np.mean(meteor_list),
        'meteor_std': np.std(meteor_list),
        'paper_hit_rate': (total - missed) * 1.0 / total,
        'review_hit_rate': (total_reviews - missed_reviews) * 1.0 / total_reviews if 'iclr' in dataset else 0.0,
        'decision_hit_rate': (total_decisions - missed_decisions) * 1.0 / total_decisions,
        'review_mae_mean': np.mean(mae_list) if 'iclr' in dataset else 0.0,
        'review_mae_std': np.std(mae_list) if 'iclr' in dataset else 0.0,
        'decision_f1': f1_score(gt_decision_lst, pr_decision_lst)
    }