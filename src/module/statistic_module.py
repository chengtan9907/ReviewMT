import glob
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os

tokenizer = None

def init_tokenizer():
    global tokenizer
    os.environ['https_proxy'] = 'http://wolfcave.myds.me:17658'
    os.environ['http_proxy'] = 'http://wolfcave.myds.me:17658'
    os.environ['all_proxy'] = 'socks5://wolfcave.myds.me:17659'
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")

def calculate_tokens(text):
    global tokenizer
    if tokenizer is None:
        raise ValueError("Tokenizer has not been initialized.")
    tokens = tokenizer.encode(text)
    return len(tokens)

def process_single_file(path):
    try:
        with open(path, 'r') as fp:
            content = json.load(fp)
        tokens = calculate_tokens(str(content))
        papers = 1
        reviews = len(content['history']) // 3
        return (papers, reviews, tokens)
    except Exception as e:
        raise Exception(f"{path} raise an error {e}")

def cal_statistic(years, path_list, year_index):
    statis = {}
    for i in year_index:
        statis[i] = {'papers': 0, 'reviews': 0, 'tokens': 0}

    print("Start statistic submission")

    with ProcessPoolExecutor(max_workers=mp.cpu_count()//2, initializer=init_tokenizer) as executor:
        futures = {executor.submit(process_single_file, path): index for index, path in enumerate(path_list)}

        results = []
        for future in tqdm(futures, total=len(futures), position=1, desc="calculating statistic information"):
            result = future.result()
            results.append(result)
    
    print("statistic calculated successfully!")

    for index, (paper, review, token) in enumerate(results):
        year = years[index]
        statis[int(year)]['papers'] += paper
        statis[int(year)]['reviews'] += review
        statis[int(year)]['tokens'] += token    
    return statis