import re
import gc
import os
import time
import json
import torch
import asyncio
import logging
import argparse
from tqdm.asyncio import tqdm_asyncio
from llamafactory.chat import chat_model
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
import glob
from tqdm import tqdm
import os.path as osp
mp.set_start_method('spawn')

def load_model(args):
    global model
    if model is None:
        model = chat_model.ChatModel(args)

def torch_gc() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

async def process_entry_async(t, index, full_context):
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
        reply = await model.achat([{"role": "user", "content": context}])
        chat_reply = reply[0].response_text
        conversation_history = [
            {"role": "user", "content": initial_prompt},
            {"role": "assistant", "content": chat_reply}
        ]
        logger.info(f"Chat time for context: {time.time() - s_time} seconds")

        for h in t['history']:
            if "author" in h[0]:
                role = 'author'
            else:
                role = "reviewer"
            roles.append(role)

            s_time = time.time()

            conversation_history.append({"role": "user", "content": h[0]})
            reply = await model.achat(conversation_history)
            chat_reply = reply[0].response_text
            conversation_history.append({"role": "assistant", "content": chat_reply})
            logger.info(f"Chat time for history: {time.time() - s_time} seconds")

            pred_replies.append(chat_reply)
            gt_replies.append(h[1])

        s_time = time.time()
        decision_prompt = 'Role: Decision Maker. Task: Suggest Accept or Reject for this paper, and provide reasons.'
        conversation_history.append({"role": "user", "content": decision_prompt})
        reply = await model.achat(conversation_history)
        chat_reply = reply[0].response_text
        logger.info(f"Chat time for instruction: {time.time() - s_time} seconds")

        roles.append("decision maker")
        pred_replies.append(chat_reply)
        gt_replies.append(t['output'])
    except Exception as e:
        logger.error(f"Error chat: {e}")

    result = {
        "title_abs": title_abs,
        "roles": roles,
        "gt_replies": gt_replies,
        "pred_replies": pred_replies,
    }

    file_path = os.path.join(output_dir, f"{index:04d}.json")
    with open(file_path, 'w', encoding='utf-8') as fp:
        json.dump(result, fp)

    torch_gc()

def process_entry(args, t, index, full_context):
    # global model
    # if model is None:
    #     load_model(args)
    load_model(args)
    
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(process_entry_async(t, index, full_context))
    return result

async def main():
    test_data = []
    if type_1 == 'test':
        with open(r"datasets/reviewmt_test.json", 'r', encoding='utf-8') as fp:
            test_data = json.load(fp)
    elif type_1 == 'train':
        path = glob.glob(r"datasets/reviewmt_train/**.json")
        for p in path:
            with open(p, 'r', encoding='utf-8') as fp:
                test_data += json.load(fp)

    max_number_of_inference = len(test_data)
    number_of_inference = 100
    if number_of_inference > max_number_of_inference:
        number_of_inference = max_number_of_inference

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with ProcessPoolExecutor(max_workers=4, initializer=load_model, initargs=(args,)) as executor:
        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(executor, process_entry, args, t, index, full_context)
            for index, t in enumerate(test_data[:number_of_inference])
        ]

        results = []
        for f in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Processing"):
            result = await f
            results.append(result)



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

full_context_list = {
    "llama3": True,
    "qwen": True,
    "baichuan2": True,
    "gemma": True,
    "deepseek": True,
    "yuan2": False,
    "chatglm3": False,
    "falcon": False,
    "yi_1.5": True,
    "glm4": False,
    "qwen2": False,
    "gemma2": True
}

model_types = ['sft', 'dpo', 'raw']
datasets_types = ['test', 'train']

os.environ["TOKENIZERS_PARALLELISM"] = "false"

global logger
global output_dir
global full_context
global args
global number_of_inference
global type_1
global arguments
global model
model = None

parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, nargs='+', default=['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'falcon', 'yi_1.5', 'glm4', 'qwen2', 'gemma2'],
                    help='The path of config file.')
parser.add_argument('--type_model', '-t1', type=str, nargs='+', default=['sft', 'dpo', 'raw'],
                    help='The type of test dataset.')
parser.add_argument('--type_data', '-t2', type=str, nargs='+', default=['test', 'train'],
                    help='The type of test dataset.')
parser.add_argument('--number', '-n', type=int, default=100,
                    help='The number of papers from test dataset to inference.')
arguments = parser.parse_args()
choose_model_list = arguments.models
for model in choose_model_list:
    if not model in models_list:
        raise Exception("Unknown model_name")
type_model = arguments.type_model
type_data = arguments.type_data
for t in type_model:
    if not t in model_types:
        raise Exception("Unknown type of model")
for t in type_data:
    if not t in datasets_types:
        raise Exception("Unknown type of dataset")
for t in type_model:
    if t == 'dpo':
        t = 'DPO'
    elif t == 'sft':
        t = 'SFT'
number_of_inference = arguments.number
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
model_list = {}
for model_name in choose_model_list:
    model_list[model_name] = {
        "model_name_or_path": f"models/{type[0]}/{models_list[model_name]}",
        "template": f"{template_list[model_name]}",
        "full_context": f"{full_context_list[model_name]}"
    }
model = None


number_of_inference = arguments.number
for model_name in tqdm(choose_model_list, position=0, desc="model"):
    for type1 in tqdm(type_model, position=1, desc="type_model"):
        for type2 in tqdm(type_data, position=2, desc="type_data"):
            type_1 = type2
            n = models_list[model_name].split("/")[-1]
            output_dir = os.path.join(r"./results/inference_results", f"{n}_{type1}_{type2}")
            full_context = full_context_list[model_name]

            name = models_list[model_name].split("/")[-1]
            if type1 == 'raw':
                model_path = osp.join(r"./models/raw", name)
                adapter_path = None
            elif type1 == 'sft':
                model_path = osp.join(r"./models/raw", name)
                adapter_path = osp.join(r"./models/SFT", name)
            elif type1 == 'dpo':
                model_path = osp.join(r"./models/raw", name)
                adapter_path = osp.join(r"./models/DPO", name)
            else:
                raise Exception(f"unknown type argument {name}")

            args = {
                "model_name_or_path": model_path,
                "adapter_name_or_path": adapter_path,
                "template": model_list[model_name]['template'],
                "max_new_tokens": 512
            }
            asyncio.run(main())


