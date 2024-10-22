import re
import gc
import os
import time
import json
import torch
import asyncio
import logging
import argparse
from tqdm import tqdm
from llamafactory.chat import chat_model
from configs.model_config import models_list, template_list
from concurrent.futures import ProcessPoolExecutor, as_completed
import os.path as osp
import shutil

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

testdataset_path = r"./datasets/reviewmt_test.json"
model = None

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

def load_model(args):
    global model
    if model is None:
        model = chat_model.ChatModel(args)

def torch_gc() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

async def process_entry_async(t, index, full_context, output_dir):
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
        logger.info(f"Chat time for {index:02} context: {time.time() - s_time} seconds")
        s_time = time.time()
        for h in t['history']:
            if "author" in h[0]:
                role = 'author'
            else:
                role = "reviewer"
            roles.append(role)

            conversation_history.append({"role": "user", "content": h[0]})
            reply = await model.achat(conversation_history)
            chat_reply = reply[0].response_text
            conversation_history.append({"role": "assistant", "content": chat_reply})

            pred_replies.append(chat_reply)
            gt_replies.append(h[1])
        logger.info(f"Chat time for {index:02} history: {time.time() - s_time} seconds")

        s_time = time.time()
        decision_prompt = 'Role: Decision Maker. Task: Suggest Accept or Reject for this paper, and provide reasons.'
        conversation_history.append({"role": "user", "content": decision_prompt})
        reply = await model.achat(conversation_history)
        chat_reply = reply[0].response_text
        logger.info(f"Chat time for {index:02} instruction: {time.time() - s_time} seconds")

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
    
    torch.gc()

def run_process_entry_async(t, index, full_context, output_dir, args):
    global model
    load_model(args)
    asyncio.run(process_entry_async(t, index, full_context, output_dir))

def main_loop(args, output_dir, full_context, number_of_inference, workers):
    with open(testdataset_path, 'r', encoding='utf-8') as fp:
        test_data = json.load(fp)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for index, t in enumerate(test_data[:number_of_inference]):
            futures.append(
                executor.submit(
                    run_process_entry_async, t, index, full_context, output_dir, args
                )
            )
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                f.result()
            except Exception as e:
                logger.error(f"Error in processing: {e}")

def main():
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument("--models", nargs='+', default=['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'falcon', 'yi_1.5', 'glm4', 'qwen2', 'gemma2'], 
                        help="The models prepared to inference")
    parser.add_argument('--type_model', '-t', type=str, nargs='+', default=['sft', 'dpo', 'raw'],
                    help='The type of model to test.')
    parser.add_argument("--workers", '-w', type=int, default=6)
    parser.add_argument('--number_of_inference', '-n', type=int, default=100,
                    help='The number of papers from test dataset to inference.')
    args = parser.parse_args()
    
    choose_model = args.models
    for i in choose_model:
        if i not in models_list:
            raise Exception("unknown model name")
    types = args.type_model
    for i in types:
        if i not in ['sft', 'dpo', 'raw']:
            raise Exception("unknown type name")
    
    with open(testdataset_path, 'r', encoding='utf-8') as fp:
        test_data = json.load(fp)
    number_of_inference = args.number_of_inference
    max_number_of_inference = len(test_data)
    if number_of_inference > max_number_of_inference:
        number_of_inference = max_number_of_inference
    
    workers = args.workers
    
    for model_name in tqdm(choose_model, desc="Inference model"):
        for type in types:
            name = models_list[model_name].split("/")[-1]
            columns = os.get_terminal_size().columns
            show_str = name + " " + type
            print("-" * (columns//2-len(show_str)), show_str, "-" * (columns//2-len(show_str)))
            
            if type == 'raw':
                model_path = osp.join(r"./models/raw", name)
                adapter_path = None
            elif type == 'sft':
                model_path = osp.join(r"./models/raw", name)
                adapter_path = osp.join(r"./models/SFT", f"{name}")
            elif type == 'dpo':
                model_path = osp.join(r"./models/raw", name)
                adapter_path = osp.join(r"./models/DPO", name)
            else:
                raise Exception(f"unknown type argument {name}")
            args_dict = {
                "model_name_or_path": model_path,
                "adapter_name_or_path": adapter_path,
                "template": template_list[model_name],
                "max_new_tokens": 512,
                "rope_scaling": "dynamic"
            }
            output_dir = osp.join(r"./results/inference_results", f"{name}_{type}")
            if osp.exists(output_dir) and os.listdir(output_dir) != 0:
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            main_loop(args=args_dict, output_dir=output_dir, full_context=FULL_CONTEXT[model_name], number_of_inference=number_of_inference, workers=workers)
        
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.set_default_device('cuda')
    main()
