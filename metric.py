import glob
from tqdm import tqdm
from src.module.metric_module import process_files
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

models_list = {
    "llama3": "NousResearch/Meta-Llama-3-8B",
    "qwen": "Qwen/Qwen-7B",
    "baichuan2": "baichuan-inc/Baichuan2-7B-Base",
    "gemma": "google/gemma-7b",
    "deepseek": "deepseek-ai/deepseek-llm-7b-base",
    "yuan2": "IEITYuan/Yuan2-2B-hf",
    "chatglm3": "THUDM/chatglm3-6b-base",
    "falcon": "tiiuae/falcon-7b",
    "yi_1.5": "01-ai/Yi-1.5-6B-Chat",
    "glm4": "THUDM/glm-4-9b",
    "qwen2": "Qwen/Qwen2-7B",
    "gemma2": "google/gemma-2-9b"
}

def process_single(path, name):
    results = process_files(path, 'iclr', len(path))
    print(f"result for {name}:")
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
    return (results, name)

def main():
    parser = argparse.ArgumentParser("metric")
    parser.add_argument("--models", "-m", nargs='+', default=['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'falcon', 'yi_1.5', 'glm4', 'qwen2', 'gemma2'])
    parser.add_argument("--types", '-t')
    args = parser.parse_args()
    choose_models = args.models
    types = args.types
    for i in choose_models:
        if not i in models_list:
            raise Exception("Unknown models")
    
    for model in tqdm(choose_models):
        with ProcessPoolExecutor(max_workers=2) as exec:
            futures = []
            for t in types:
                modelPath_with_size = []
                name = models_list[model].split("/")[-1] + f"_{t}"
                paths = glob.glob(fr"results/inference_results/{name}**/**.json")
                paths.sort()
                
                for p in paths:
                    modelPath_with_size.append(((os.path.getsize(p)), p))
                modelPath_with_size.sort(key=lambda x:x[0], reverse=True)
                models_to_deal = [i[1] for i in modelPath_with_size[0:100]]

                futures.append(exec.submit(process_single, models_to_deal, name))
                
            for future in as_completed(futures):
                result = future.result()
                name = result[1]
                result = result[0]
                if not result:
                    raise Exception("Metric error")
                else:
                    with open(fr"results/metric_results/{name}.txt", 'w') as fp:
                        fp.write(f"BLEU-2 Mean: {result['bleu2_mean']:.2f}, Std: {result['bleu2_std']:.2f}\n")
                        fp.write(f"BLEU-4 Mean: {result['bleu4_mean']:.2f}, Std: {result['bleu4_std']:.2f}\n")
                        fp.write(f"BERT Mean: {result['bert_mean']:.2f}, Std: {result['bert_std']:.2f}\n")
                        fp.write(f"ROUGE-L Mean: {result['rouge_l_mean']:.2f}, Std: {result['rouge_l_std']:.2f}\n")
                        fp.write(f"ROUGE-1 Mean: {result['rouge1_mean']:.2f}, Std: {result['rouge1_std']:.2f}\n")
                        fp.write(f"ROUGE-2 Mean: {result['rouge2_mean']:.2f}, Std: {result['rouge2_std']:.2f}\n")
                        fp.write(f"Meteor Mean: {result['meteor_mean']:.2f}, Std: {result['meteor_std']:.2f}\n")
                        fp.write(f"Paper Hit Rate: {result['paper_hit_rate']:.4f}\n")
                        fp.write(f"Review Hit Rate: {result['review_hit_rate']:.4f}\n")
                        fp.write(f"Decision Hit Rate: {result['decision_hit_rate']:.4f}\n")
                        fp.write(f"Review MAE MEAN: {result['review_mae_mean']:.2f}, Std: {result['review_mae_std']:.2f}\n")
                        fp.write(f"Decision F1 MEAN: {result['decision_f1']:.4f}\n")
            
if __name__ == "__main__":
    main()
    # path = glob.glob(r"results/inference_results/new_sft/done/gpt4o_raw_test/**.json")
    # process_single(path, "gpt4o_raw")

# llama3 qwen2 yi_1.5 falcon glm4 baichuan2 
# qwen gemma deepseek yuan2 chatglm3 gemma2