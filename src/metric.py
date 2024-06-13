import os
import json
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import re
from sklearn.metrics import f1_score
import argparse

# Initialize scorers
bleu_smoothie = SmoothingFunction().method4
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
parser = argparse.ArgumentParser(description='Make datasets for iclr')

parser.add_argument("path2yaml", type=str, help="Path to the arguments yaml")


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
        nltk.download('wordnet')
        score = meteor_score([reference], hypothesis)
    return score

def process_files(path, dataset='iclr', total=100):    
    bleu2_list, bleu4_list, rouge_l_list, rouge1_list, rouge2_list, meteor_list = [], [], [], [], [], []
    mae_list = []
    gt_decision_lst, pr_decision_lst = [], []
    missed = 0
    missed_reviews, total_reviews = 0, 0
    missed_decisions, total_decisions = 0, 0

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
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
            bleu4, bleu2 = calculate_bleu(gt_reply.split(), pr_reply.split())
            rouge_l, rouge1, rouge2 = calculate_rouge(gt_reply, pr_reply)
            meteor = calculate_meteor(gt_reply.split(), pr_reply.split())

            # reviewer score
            if role == 'reviewer' and (idx + 1) % 3 == 0:
                total_reviews += 1

                gt_score_pattern = r"score: (\d+)"
                gt_score = int(re.findall(gt_score_pattern, gt_reply)[0])
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

def main():
    if not os.path.exists(yaml_path):
        print("The path of yaml is not exists")
        raise Exception("path not exists")
    yaml_path = parser.parse_args().path2yaml
    with open(yaml_path, 'r') as fp:
        y = yaml.safe_load(fp)

    if 'datasets' in y:
        datasets = y['datasets']
    else:
        raise Exception("format wrong in config file")
    if 'model_names' in y:
        model_names = y['model_names']
    else:
        raise Exception("format wrong in config file")
    if 'total_papers' in y:
        total_papers = y['total_papers']
    else:
        raise Exception("format wrong in config file")
    const_datasets = ['finetune_iclr', 'finetune_nc', 'raw_iclr', 'raw_nc']
    const_model_names = ['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'yi', 'falcon', 'glm4', 'qwen2']
    for dataset in datasets:
        if not dataset in const_datasets:
            continue
        print('---------- ' + dataset + ' ----------')
        for model_name in model_names:
            if not model_name in const_model_names:
                continue
            path = f'data/test_results/{dataset}/{model_name}_test'
            # path = f'{model_name}_test'
            if not os.path.exists(path):
                print(f"Path {path} does not exist.\n")
                continue
            results = process_files(path, dataset, total_papers)
            print(f"Results for {model_name}:")
            print(f"BLEU-2 Mean: {results['bleu2_mean']:.2f}, Std: {results['bleu2_std']:.2f}")
            print(f"BLEU-4 Mean: {results['bleu4_mean']:.2f}, Std: {results['bleu4_std']:.2f}")
            print(f"ROUGE-L Mean: {results['rouge_l_mean']:.2f}, Std: {results['rouge_l_std']:.2f}")
            print(f"ROUGE-1 Mean: {results['rouge1_mean']:.2f}, Std: {results['rouge1_std']:.2f}")
            print(f"ROUGE-2 Mean: {results['rouge2_mean']:.2f}, Std: {results['rouge2_std']:.2f}")
            print(f"Meteor Mean: {results['meteor_mean']:.2f}, Std: {results['meteor_std']:.2f}")
            print(f"Paper Hit Rate: {results['paper_hit_rate']:.4f} ")
            if 'iclr' in dataset:
                print(f"Review Hit Rate: {results['review_hit_rate']:.4f} ")
            print(f"Decision Hit Rate: {results['decision_hit_rate']:.4f} ")
            if 'iclr' in dataset:
                print(f"Review MAE MEAN: {results['review_mae_mean']:.2f}, Std: {results['review_mae_std']:.2f} ")
            print(f"Decision F1 MEAN: {results['decision_f1']:.4f}\n")

if __name__ == "__main__":
    main()
