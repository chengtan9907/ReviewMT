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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='qwen',
                    help='The path of config file.')
parser.add_argument('--number', '-n', type=int, default=100,
                    help='The number of papers from test dataset to inference.')

arguments = parser.parse_args()

model_name = arguments.model


number_of_inference = arguments.number

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

output_dir = r"./results/inference_results/Meta-Llama-3-8B_sft_test"

model_list = {
    'llama3': {
        "model_name_or_path": "./models/raw/Meta-Llama-3-8B",
        "adapter_name_or_path": "./models/SFT/Meta-Llama-3-8B",
        "template": "llama3",
        "full_context": True
    },
    'yuan2': {
        "model_name_or_path": "data/models/Yuan2-2B-hf",
        "template": "yuan",
        "full_context": False
    },
    'baichuan2': {
        "model_name_or_path": "data/models/Baichuan2-7B-Base",
        "template": "baichuan2",
        "full_context": True
    },
    'chatglm3': {
        "model_name_or_path": "data/models/chatglm3-6b-base",
        "template": "chatglm3",
        "full_context": False
    },
    'deepseek': {
        "model_name_or_path": "data/models/deepseek-llm-7b-base",
        "template": "deepseek",
        "full_context": True
    },
    'gemma': {
        "model_name_or_path": "data/models/gemma-7b",
        "template": "gemma",
        "full_context": True
    },
    'qwen': {
        "model_name_or_path": "data/models/Qwen-7B",
        "template": "qwen",
        "full_context": True
    },
    'falcon': {
        "model_name_or_path": "data/models/falcon-7b",
        "template": "falcon",
        "full_context": False
    },
    'yi': {
        "model_name_or_path": "data/models/Yi-1.5-6B-Chat",
        "template": "yi",
        "full_context": True
    },
    'glm4': {
        "model_name_or_path": "data/models/glm-4-9b",
        "template": "glm4",
        "full_context": False
    },
    'qwen2': {
        "model_name_or_path": "data/models/Qwen2-7B",
        "template": "qwen",
        "full_context": False
    }
}

args = {
    "model_name_or_path": model_list[model_name]['model_name_or_path'],
    "adapter_name_or_path": model_list[model_name]['adapter_name_or_path'],
    "template": model_list[model_name]['template'],
    "max_new_tokens": 512
}

model = None
full_context = model_list[model_name]['full_context']

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
    with open(r"./datasets/reviewmt_test.json", 'r', encoding='utf-8') as fp:
        test_data = json.load(fp)

    # max_number_of_inference = len(test_data)
    
    # if number_of_inference > max_number_of_inference:
    #     number_of_inference = max_number_of_inference
    
    number_of_inference = 100

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

if __name__ == '__main__':
    mp.set_start_method('spawn')
    asyncio.run(main())