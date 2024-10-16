from tqdm import tqdm
import subprocess
import glob
import os
import argparse
import json

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

template_list = {
    "llama3": "llama3",
    "qwen": "qwen",
    "baichuan2": "baichuan2",
    "gemma": "gemma",
    "deepseek": "deepseek",
    "yuan2": "yuan",
    "chatglm3": "chatglm3",
    "falcon": "falcon",
    "yi_1.5": "yi",
    "glm4": "glm4",
    "qwen2": "qwen",
    "gemma2": "gemma"
}

def make_datasets():
    data = glob.glob(r"./data/converted/DPO/**.json")
    data.sort()
    results = []
    for d in data:
        with open(d, 'r') as fp:
            content = json.load(fp)
        results.append(content)
    with open(r"./datasets/reviewmt_dpo.json", 'w') as fp:
        json.dump(results, fp)

def main():
    parser = argparse.ArgumentParser("DPO")
    parser.add_argument("--models", "-m", nargs='+', default=['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'falcon', 'yi_1.5', 'glm4', 'qwen2', 'gemma2'], help="base models prepared to finetune on, choose from [llama3, qwen, baichuan2, gemma, deepseek, yuan2, chatglm3, falcon, yi_1.5, glm4, qwen2, gemma2]")
    parser.add_argument("--batchsize", "-b", default=1, help="train batch size on per device (default: 1)")
    args = parser.parse_args()
    choose_models_list = args.models
    make_datasets()

    print("All datasets have been made succesfully.")
    BATCH_SIZE = args.batchsize
    for model in tqdm(choose_models_list, desc="Model DPO", total=len(choose_models_list)):
        name = models_list[model].split("/")[-1]
        cmd = [
            "llamafactory-cli", "train",
            "--model_name_or_path", os.path.join(r"./models/raw", models_list[model].split("/")[-1]),
            "--adapter_name_or_path", os.path.join(r"./models/SFT", models_list[model].split("/")[-1]),
            "--stage", "dpo",
            "--do_train", "true",
            "--finetuning_type", "lora",
            "--lora_target", "all",
            "--pref_beta", "0.1",
            "--pref_loss", "sigmoid",  # choices: [sigmoid (dpo), orpo, simpo]
            "--dataset_dir", "datasets",
            "--dataset", "ReviewMT_DPO",
            "--template", template_list[model],
            "--cutoff_len", "1024",
            "--overwrite_cache", "true",
            "--preprocessing_num_workers", "1",
            "--output_dir", f"models/DPO/{name}",
            "--logging_steps", "10",
            "--save_steps", "500",
            "--plot_loss", "true",
            "--overwrite_output_dir", "true",
            "--per_device_train_batch_size", f"{BATCH_SIZE}",
            "--gradient_accumulation_steps", "8",
            "--learning_rate", "5.0e-6",
            "--num_train_epochs", "3.0",
            "--lr_scheduler_type", "cosine",
            "--warmup_ratio", "0.1",
            "--bf16", "true",
            "--ddp_timeout", "180000000",
            "--val_size", "0.1",
            "--per_device_eval_batch_size", "1",
            "--eval_strategy", "steps",
            "--eval_steps", "500",
            "--rope_scaling", "dynamic",
            "--cache_dir", "./DPOcache"
        ]
        subprocess.run(cmd)

if __name__ == "__main__":
    main()