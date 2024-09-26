from llmtuner.chat import chat_model
import argparse
from tqdm import tqdm
import os
import os.path as osp
import json
import time
import re
import gc
import torch
import torch.multiprocessing as mp
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.chdir(r"/tancheng/lvdx/ReviewMT_plus")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

datasets = {
    'train': r"./datasets/reviewmt_train",
    'test': r"./datasets/reviewmt_test.json"
}

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

FULL_CONTEXT = {
    "llama3": False,
    "qwen": True,
    "baichuan2": True,
    "gemma": True,
    "deepseek": True,
    "yuan2": False,
    "chatglm3": True,
    "falcon": False,
    "yi_1.5": True,
    "glm4": False,
    "qwen2": False,
    "gemma2": True
}

def remove_unicode_sequences(input_string):
    if isinstance(input_string, list):
        for i in input_string:
            i = re.sub(r'\\u[0-9a-fA-F]{4}', '', i)
        return input_string
    else:
        return re.sub(r'\\u[0-9a-fA-F]{4}', '', input_string)

# inference single model with TYPE(raw, sft, dpo) MODEL_NAME on datasets(a file or a path)
def inference(model_name, datasets, type, number_of_inference=None):
    name = models_list[model_name].split("/")[-1]
    if type == 'raw':
        model_path = osp.join(r"./models/raw", name)
        adapter_path = None
    elif type == 'sft':
        model_path = osp.join(r"./models/raw", name)
        adapter_path = osp.join(r"./models/new_sft", f"{name}")
    elif type == 'dpo':
        model_path = osp.join(r"./models/raw", name)
        adapter_path = osp.join(r"./models/DPO", name)
    else:
        raise Exception(f"unknown type argument {name}")

    test_data = []

    if osp.isdir(datasets):
        root_path = datasets
        data = os.listdir(datasets)
        data = [osp.join(root_path, i) for i in data]
        data.sort()
        for d in data:
            with open(d, 'r') as fp:
                test_data += json.load(fp)
    elif osp.isfile(datasets):
        with open(datasets, 'r') as fp:
            test_data += json.load(fp)
    else:
        raise Exception("unrecognized datasets")

    max_number_of_inference = len(test_data)
    if number_of_inference is None or number_of_inference > max_number_of_inference:
        number_of_inference = max_number_of_inference

    args = {
        "model_name_or_path": model_path,
        "adapter_name_or_path": adapter_path,
        "template": template_list[model_name],
        "max_new_tokens": 512,
        "rope_scaling": "dynamic"
    }

    dataset_type = "train" if "train" in datasets else "test"
    output_dir = osp.join(r"./results/inference_results/new_sft", f"{name}_{type}_{dataset_type}")
    os.makedirs(output_dir, exist_ok=True)

    # Limit the number of processes to the number of available GPUs
    num_gpus = torch.cuda.device_count()
    processes = min(num_gpus, 5)  # Assuming a maximum of 4 processes

    ctx = mp.get_context("spawn")  # Ensures that CUDA works well with multiprocesses
    pool = ctx.Pool(processes=4, initializer=load_model, initargs=(args,))
    
    results = []
    futures = [
        pool.apply_async(process_entry_async, args=(t, index, FULL_CONTEXT[model_name], output_dir, f"{name}_{type}_{dataset_type}"))
        for index, t in enumerate(test_data[:number_of_inference])
    ]

    for future in tqdm(futures, total=len(futures), desc="Processing"):
        try:
            result = future.get()
            results.append(result)
        except Exception as e:
            print(f"Error during inference: {e}")

    pool.close()
    pool.join()

def load_model(args):
    global model
    # device_id = (mp.current_process()._identity[0] - 1) % torch.cuda.device_count()
    # torch.cuda.set_device(device_id)
    model = chat_model.ChatModel(args)
    return model

def process_entry_async(t, index, full_context, output_dir, type_train):
    global model
    pattern = r"Title:\s(.*?)\sAbstract:\s.*?\."
    try:
        match = re.search(pattern, t['input'])
        title_abs = match.group(0)
    except Exception as e:
        logger.error(f"Error extracting title and abstract: {e}")
        title_abs = ''

    initial_prompt = "This is a peer-review system. You will be assigned with roles such as author, reviewer or decision maker to perform different tasks. "
    context = initial_prompt + "Please summarize and remember this paper: "
    context = (context + title_abs) if not full_context else (context + t['input'])

    gt_replies, pred_replies, roles = [], [], []

    try:
        s_time = time.time()
        with torch.no_grad():
            reply = model.chat([{"role": "user", "content": context}])
        chat_reply = reply[0].response_text
        conversation_history = [
            {"role": "user", "content": initial_prompt},
            {"role": "assistant", "content": chat_reply}
        ]
        logger.info(f"{type_train} Chat time for {index:04} context: {time.time() - s_time} seconds")
        role = None
        s_time = time.time()
        for idx, h in enumerate(t['history']):
            if "author" in h[0]:
                role = 'authors'
            else:
                role = "reviewer"
            roles.append(role)

            conversation_history.append({"role": "user", "content": h[0]})
            with torch.no_grad():
                reply = model.chat(conversation_history)
            torch_gc()
            # print()
            # print("-"*os.get_terminal_size().columns)
            # print(f"{index:04} history: {idx}/{len(t['history'])}")
            # gpu_info()
            chat_reply = reply[0].response_text
            conversation_history.append({"role": "assistant", "content": chat_reply})

            pred_replies.append(chat_reply)
            gt_replies.append(h[1])
            
        logger.info(f"{type_train} Chat time for {index:04} history: {time.time() - s_time} seconds")

        s_time = time.time()
        decision_prompt = 'Role: Decision Maker. Task: Suggest Accept or Reject for this paper, and provide reasons.'
        conversation_history.append({"role": "user", "content": decision_prompt})
        with torch.no_grad():
            reply = model.chat(conversation_history)
        
        chat_reply = reply[0].response_text
        logger.info(f"{type_train} Chat time for instruction: {time.time() - s_time} seconds")
        logger.info("")

        roles.append("decision maker")
        pred_replies.append(chat_reply)
        gt_replies.append(t['output'])
    except Exception as e:
        logger.error(f"Error during chat: {e}")

    result = {
        "title_abs": title_abs,
        "roles": roles,
        "gt_replies": gt_replies,
        "pred_replies": remove_unicode_sequences(pred_replies),
    }
    
    
    file_path = os.path.join(output_dir, f"{index:04d}.json")
    with open(file_path, 'w', encoding='utf-8') as fp:
        json.dump(result, fp)
    torch_gc()

# def gpu_info():
#     current_gpu_index = torch.cuda.current_device()
#     total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory
#     allocated_memory = torch.cuda.memory_allocated(current_gpu_index)
#     free_memory = total_memory - allocated_memory
#     total_memory_MB = total_memory / (1024 ** 2)
#     allocated_memory_MB = allocated_memory / (1024 ** 2)
#     free_memory_MB = free_memory / (1024 ** 2)
    
#     print(f"Current GPU index: \t{current_gpu_index}")
#     print(f"Total GPU memory: \t{total_memory_MB:.2f} MB")
#     print(f"Allocated GPU memory: \t {allocated_memory_MB:.2f} MB")
#     print(f"Free GPU memory: \t{free_memory_MB:.2f} MB")
#     print("-"*os.get_terminal_size().columns)
#     print()

def torch_gc() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def main():
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument("--models", nargs='+', default=['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'falcon', 'yi_1.5', 'glm4', 'qwen2', 'gemma2'], help="The models prepared to inference")
    parser.add_argument('--type_model', '-t1', type=str, nargs='+', default=['sft', 'dpo', 'raw'],
                    help='The type of test dataset.')
    args = parser.parse_args()
    choose_model = args.models
    for i in choose_model:
        if not i in models_list:
            raise Exception("unknown model name")

    for model in tqdm(choose_model, desc="Inference model"):
        for dataset in datasets.values():
            for type in args.type_model:
                columns = os.get_terminal_size().columns
                show_str = model + " " + type + " " + dataset.split("/")[-1]
                print("-" * (columns//2-len(show_str)), show_str, "-" * (columns//2-len(show_str)))
                inference(model, dataset, type, 100)  # Assuming we want to limit the number of inferences

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Use 'spawn' start method
    main()
