import subprocess
import os
import argparse
from configs.model_config import models_list, template_list
from huggingface_hub import snapshot_download

def download_model(model_name, save_path):
    times = 0
    force_download = True
    while True:
        try:
            snapshot_download(repo_id=model_name, local_dir=save_path, force_download=force_download, proxies=None, resume_download=True, token="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")
            break
        except Exception:
            times += 1
            if times <= 20:
                print()
                print('-'*os.get_terminal_size().columns)
                print(f"Retry again {times}/20")
                print()
                force_download = False
                continue
            else:
                raise Exception("Too many retry times!")

def main():
    parser = argparse.ArgumentParser("SFT")
    parser.add_argument("--models", nargs='+', default=['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'falcon', 'yi_1.5', 'glm4', 'qwen2', 'gemma2'], help="base models prepared to finetune on, choose from [llama3, qwen, baichuan2, gemma, deepseek, yuan2, chatglm3, falcon, yi_1.5, glm4, qwen2, gemma2]")
    parser.add_argument("--batch_size", default=2, help="batch size on per device")
    args = parser.parse_args()

    choose_models_list = args.models
    for i in choose_models_list:
        if not i in models_list:
            raise Exception("An unknown model is chosen")
    print("Choose Models: ", choose_models_list)

    for key, model in models_list.items():
        if not key in choose_models_list:
            continue
        name = model.split("/")[-1]
        base_model_path = r"./models/raw"
        if os.path.exists(os.path.join(base_model_path, name)):
            y = input(f"{key} is already exists, whether to force download to update it? [y/[n]]")
            if not y.lower() == 'y':
                continue
        print(f"Start to download {name}")
        download_model(model, os.path.join(base_model_path, name))
    
    choice = input("All models have been downloaded, shall we continue? [y]/n")
    if choice.lower() == 'n':
        print("Program determination.")
        exit()
    
    #use llamafactory to train
    BATCH_SIZE = args.batch_size
    for model in choose_models_list:
        name = models_list[model].split("/")[-1]
        cmd = [
            "llamafactory-cli", "train",
            "--model_name_or_path", os.path.join(r"./models/raw", models_list[model].split("/")[-1]),
            "--stage", "sft",
            "--do_train", "true",
            "--finetuning_type", "lora",
            "--lora_target", "all",
            "--dataset_dir", "datasets",
            "--dataset", "alpaca_en_demo,ReviewMT",
            "--template", template_list[model],
            "--cutoff_len", "1024",
            "--overwrite_cache", "true",
            "--preprocessing_num_workers", "16",
            "--output_dir", f"models/SFT/{name}",
            "--logging_steps", "10",
            "--save_steps", "500",
            "--plot_loss", "true",
            "--overwrite_output_dir", "true",
            "--per_device_train_batch_size", f"{BATCH_SIZE}",
            "--gradient_accumulation_steps", "8",
            "--learning_rate", "1.0e-5",
            "--num_train_epochs", "3.0",
            "--lr_scheduler_type", "cosine",
            "--warmup_ratio", "0.1",
            "--bf16", "true",
            "--ddp_timeout", "180000000",
            "--val_size", "0.1",
            "--per_device_eval_batch_size", "1",
            "--eval_strategy", "steps",
            "--eval_steps", "500",
            "--cache_dir", "./cache",
            "--rope_scaling", "dynamic"
        ]
        subprocess.run(cmd)
        print()

if __name__ == '__main__':
    main()