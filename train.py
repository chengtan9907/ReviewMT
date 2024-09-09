import subprocess
import os
import glob
import random
import argparse
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait
import json
import re
import shutil
import time
from huggingface_hub import snapshot_download
import requests

train_datasets_lock = threading.Lock()
test_datasets_lock = threading.Lock()

def download_model(model_name, save_path):
    proxies = {
        "http": "http://wolfcave.myds.me:17658",
        "https": "http://wolfcave.myds.me:17658",
        "all": "socks5://wolfcave.myds.me:17659"
    }
    times = 0
    force_download = True
    while True:
        try:
            snapshot_download(repo_id=model_name, local_dir=save_path, force_download=force_download, proxies=proxies, resume_download=True, token="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")
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

def correct_double_periods(train_datasets, test_datasets):
    start_time = time.time()
    for t in train_datasets:
        t['instruction'] = re.sub(r'(?<!\.)\.\.(?!\.)', '.', t['instruction'])
        t['input'] = re.sub(r'(?<!\.)\.\.(?!\.)', '.', t['input'])
        t['output'] = re.sub(r'(?<!\.)\.\.(?!\.)', '.', t['output'])
        for h in t['history']:
            h = [re.sub(r'(?<!\.)\.\.(?!\.)', '.', i) for i in h]
    for t in test_datasets:
        t['instruction'] = re.sub(r'(?<!\.)\.\.(?!\.)', '.', t['instruction'])
        t['input'] = re.sub(r'(?<!\.)\.\.(?!\.)', '.', t['input'])
        t['output'] = re.sub(r'(?<!\.)\.\.(?!\.)', '.', t['output'])
        for h in t['history']:
            h = [re.sub(r'(?<!\.)\.\.(?!\.)', '.', i) for i in h]
    print(f'Fixing time used: {time.time()-start_time}s')
    return train_datasets, test_datasets

def open_file(file, type):
    global train_datasets
    global test_datasets
    with open(file, 'r') as fp:
        content = json.load(fp)
    if type == 'train':
        with train_datasets_lock:
            train_datasets.append(content)
    else:
        with test_datasets_lock:
            test_datasets.append(content)

def datasetsLoad(test):
    iclr_data_path = glob.glob(r"./data/converted/ICLR/**.json")
    iclr_data_path.sort()
    nips_data_path = glob.glob(r"./data/converted/NeurIPS/**.json")
    nips_data_path.sort()
    uai_data_path = glob.glob(r"./data/converted/UAI/**.json")
    uai_data_path.sort()

    num_iclr = len(iclr_data_path)
    num_nips = len(nips_data_path)
    num_uai = len(uai_data_path)

    test = [-i for i in range(0, 7296)]
    random.shuffle(test)
    test = test[:100]

    print(f"Datasets loaded as below:")
    print(f"ICLR: {num_iclr}")
    print(f"NeurIPS: {num_nips}")
    print(f"UAI: {num_uai}")
    print(f"Using {len(test)} samples as test datasets.")
    print("-"*os.get_terminal_size().columns)
    return test, iclr_data_path, nips_data_path, uai_data_path

def save_dataset_chunks(train_datasets, chunk_size):    
    os.makedirs("./datasets/reviewmt_train", exist_ok=True)
    if len(os.listdir(r"./datasets/reviewmt_train")) != 0:
        raise Exception("reviewmt_train is not empty")
    
    split_list = [train_datasets[i:i + chunk_size] for i in range(0, len(train_datasets), chunk_size)]
    
    for index, split in tqdm(enumerate(split_list), total=len(split_list), desc="split and save train_datasets in chunks", leave=False, position=1):
        with open(fr"./datasets/reviewmt_train/{index:02}.json", 'w') as fp:
            json.dump(split, fp)


def main():
    parser = argparse.ArgumentParser("SFT")
    parser.add_argument("--models", nargs='+', default=['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'falcon', 'yi_1.5', 'glm4', 'qwen2'], help="base models prepared to finetune on, choose from [llama3, qwen, baichuan2, gemma, deepseek, yuan2, chatglm3, falcon, yi_1.5, glm4, qwen2]")
    parser.add_argument("--test", default=100, help="size of test datasets (random extraction from ICLR2024)")
    parser.add_argument("--chunk_size", default=1000, help="the chunk size of train datasets split")
    parser.add_argument("--tworkers", default=100, help="number of threads dealing with datasets")

    args = parser.parse_args()
    
    global train_datasets
    train_datasets = []
    global test_datasets
    test_datasets = []
    
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
    
    choose_models_list = args.models
    for i in choose_models_list:
        if not i in models_list:
            raise Exception("An unknown model is chosen")
    print("Choose Models: ", choose_models_list)
    
    test, iclr_data_path, nips_data_path, uai_data_path = datasetsLoad(args.test)

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
    
    input("All models have been downloaded, shall we continue?")

    with ThreadPoolExecutor(max_workers=args.tworkers) as executor:
        with tqdm(total=3, desc="Load and split datasets") as tqd:
            futures = []
            for i in nips_data_path:
                future = executor.submit(open_file, i, 'train')
                futures.append(future)
            wait(futures)
            tqd.update()
            
            futures = []
            for i in uai_data_path:
                future = executor.submit(open_file, i, 'train')
                futures.append(future)
            wait(futures)
            tqd.update()

            futures = []
            test = [len(iclr_data_path)+i for i in test]
            for i in iclr_data_path:
                number = int(i.split("/")[-1].replace(".json", ""))
                if number in test:
                    future = executor.submit(open_file, i, 'test')
                    futures.append(future)
                else:
                    future = executor.submit(open_file, i, 'train')
                    futures.append(future)
            wait(futures)
            tqd.update()

    
    correct_double_periods(train_datasets, test_datasets)
    shutil.rmtree(r"./datasets/reviewmt_train")
    save_dataset_chunks(train_datasets, chunk_size=args.chunk_size)
    with open(r"./datasets/reviewmt_test.json", 'w') as fp:
        json.dump(test_datasets, fp)
    
    print()
    print("Datasets have been splited")
    print(f"Train Datasets Size: {len(train_datasets)}")
    print(f"Test Datasets Size: {len(test_datasets)}")
    print("-"*os.get_terminal_size().columns)
    input("Continue?")
    
    # use llamafactory to train
    
    BATCH_SIZE = 2
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
            "--dataset", "identity,alpaca_en_demo,alpaca_zh_demo,ReviewMT",
            "--template", template_list[model],
            "--cutoff_len", "10240",
            "--overwrite_cache", "true",
            "--preprocessing_num_workers", "16",
            "--output_dir", f"models/{name}",
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
            "--rope_scaling", "linear"
        ]
        subprocess.run(cmd)
        print()
        input("Continue?")

if __name__ == '__main__':
    main()
