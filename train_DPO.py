from tqdm import tqdm
import subprocess
import glob
import os
import argparse
import json
from configs.model_config import models_list, template_list

def main():
    parser = argparse.ArgumentParser("DPO")
    parser.add_argument("--models", "-m", nargs='+', default=['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'falcon', 'yi_1.5', 'glm4', 'qwen2', 'gemma2'], help="base models prepared to finetune on, choose from [llama3, qwen, baichuan2, gemma, deepseek, yuan2, chatglm3, falcon, yi_1.5, glm4, qwen2, gemma2]")
    parser.add_argument("--batch_size", default=1, help="train batch size on per device (default: 1)")
    args = parser.parse_args()
    choose_models_list = args.models

    BATCH_SIZE = args.batch_size
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