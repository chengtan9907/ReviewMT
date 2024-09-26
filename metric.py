import glob
from tqdm import tqdm
from src.module.metric_module import process_files
import os



# def main():
#     # if not os.path.exists(yaml_path):
#     #     print("The path of yaml is not exists")
#     #     raise Exception("path not exists")
#     # yaml_path = parser.parse_args().path2yaml
#     # with open(yaml_path, 'r') as fp:
#     #     y = yaml.safe_load(fp)

#     if 'datasets' in y:
#         datasets = y['datasets']
#     else:
#         raise Exception("format wrong in config file")
#     if 'model_names' in y:
#         model_names = y['model_names']
#     else:
#         raise Exception("format wrong in config file")
#     if 'total_papers' in y:
#         total_papers = y['total_papers']
#     else:
#         raise Exception("format wrong in config file")
#     const_datasets = ['finetune_iclr', 'finetune_nc', 'raw_iclr', 'raw_nc']
#     const_model_names = ['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'yi', 'falcon', 'glm4', 'qwen2']
#     for dataset in datasets:
#         if not dataset in const_datasets:
#             continue
#         print('---------- ' + dataset + ' ----------')
#         for model_name in model_names:
#             if not model_name in const_model_names:
#                 continue
#             path = f'data/test_results/{dataset}/{model_name}_test'
#             # path = f'{model_name}_test'
#             if not os.path.exists(path):
#                 print(f"Path {path} does not exist.\n")
#                 continue
#             results = process_files(path, dataset, total_papers)
#             print(f"Results for {model_name}:")
#             print(f"BLEU-2 Mean: {results['bleu2_mean']:.2f}, Std: {results['bleu2_std']:.2f}")
#             print(f"BLEU-4 Mean: {results['bleu4_mean']:.2f}, Std: {results['bleu4_std']:.2f}")
#             print(f"ROUGE-L Mean: {results['rouge_l_mean']:.2f}, Std: {results['rouge_l_std']:.2f}")
#             print(f"ROUGE-1 Mean: {results['rouge1_mean']:.2f}, Std: {results['rouge1_std']:.2f}")
#             print(f"ROUGE-2 Mean: {results['rouge2_mean']:.2f}, Std: {results['rouge2_std']:.2f}")
#             print(f"Meteor Mean: {results['meteor_mean']:.2f}, Std: {results['meteor_std']:.2f}")
#             print(f"Paper Hit Rate: {results['paper_hit_rate']:.4f} ")
#             if 'iclr' in dataset:
#                 print(f"Review Hit Rate: {results['review_hit_rate']:.4f} ")
#             print(f"Decision Hit Rate: {results['decision_hit_rate']:.4f} ")
#             if 'iclr' in dataset:
#                 print(f"Review MAE MEAN: {results['review_mae_mean']:.2f}, Std: {results['review_mae_std']:.2f} ")
#             print(f"Decision F1 MEAN: {results['decision_f1']:.4f}\n")

if __name__ == "__main__":
    path = glob.glob(r"results/inference_results/Meta-Llama-3-8B_sft_test_4096/**.json")
        # path = f'{model_name}_test'

    results = process_files(path, 'iclr', len(path))
    print(f"Results for Llama3:")
    print(f"BLEU-2 Mean: {results['bleu2_mean']:.2f}, Std: {results['bleu2_std']:.2f}")
    print(f"BLEU-4 Mean: {results['bleu4_mean']:.2f}, Std: {results['bleu4_std']:.2f}")
    print(f"BERT Mean: {results['bert_mean']:.2f}, Std: {results['bert_std']:.2f}")
    print(f"ROUGE-L Mean: {results['rouge_l_mean']:.2f}, Std: {results['rouge_l_std']:.2f}")
    print(f"ROUGE-1 Mean: {results['rouge1_mean']:.2f}, Std: {results['rouge1_std']:.2f}")
    print(f"ROUGE-2 Mean: {results['rouge2_mean']:.2f}, Std: {results['rouge2_std']:.2f}")
    print(f"Meteor Mean: {results['meteor_mean']:.2f}, Std: {results['meteor_std']:.2f}")
    print(f"Paper Hit Rate: {results['paper_hit_rate']:.4f} ")
    print(f"Review Hit Rate: {results['review_hit_rate']:.4f} ")
    print(f"Decision Hit Rate: {results['decision_hit_rate']:.4f} ")
    print(f"Review MAE MEAN: {results['review_mae_mean']:.2f}, Std: {results['review_mae_std']:.2f} ")
    print(f"Decision F1 MEAN: {results['decision_f1']:.4f}\n")