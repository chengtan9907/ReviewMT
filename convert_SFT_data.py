import json
import os
import torch.multiprocessing as mp
from tqdm import tqdm
import glob
import random
import shutil
import argparse
from src.module.statistic_module import cal_statistic
from src.module.readData import read_all_data
from src.module.convert_module import process_uai_data, process_iclr_data, process_nips_data, correct_double_periods

def convert_nips(nips_review_path, nips_paper_path, statistic=False):
    print("Start to convert NIPS data.")
    os.makedirs(r"./data/converted/NeurIPS", exist_ok=True)
    pool = mp.Pool(mp.cpu_count())

    results = []
    with tqdm(total=len(nips_paper_path), desc="NIPS Conversion", position=1) as pbar:
        for result in pool.imap(process_nips_data, zip(nips_paper_path, nips_review_path)):
            if result == False:
                continue
            results.append(result)
            pbar.update()

    pool.close()
    pool.join()

    for index, r in enumerate(results):
        r = correct_double_periods(r)
        with open(fr"./data/converted/NeurIPS/{index:04}.json", 'w') as fp:
            json.dump(r, fp)

    if statistic:
        path_list = [fr"./data/converted/NeurIPS/{index:04}.json" for index in range(len(results))]
        years = []
        for r in results:
            years.append(r[2])
        statis = cal_statistic(years, path_list, range(2021, 2024))
        return statis

def convert_iclr(iclr_review_path, iclr_paper_path, statistic=False):
    print("Start to convert ICLR data.")
    os.makedirs(r"./data/converted/ICLR", exist_ok=True)
    pool = mp.Pool(mp.cpu_count())
    results = []
    with tqdm(total=len(iclr_paper_path), desc="ICLR Conversion", position=1) as pbar:
        for result in pool.imap(process_iclr_data, zip(iclr_paper_path, iclr_review_path)):
            results.append(result)
            pbar.update()
    pool.close()
    pool.join()

    results = [(d, e, f) for d, e, f in results if d != False]


    for index, r in enumerate(results):
        r = correct_double_periods(r)
        with open(fr"./data/converted/ICLR/{index:05}.json", 'w') as fp:
            json.dump(r, fp)
    
    if statistic:
        path_list = [fr"./data/converted/ICLR/{index:05}.json" for index in range(len(results))]
        years = []
        for r in results:
            years.append(r[2])
        statis = cal_statistic(years, path_list, range(2017, 2025))
        return statis

def convert_uai(uai_review_path, uai_paper_path, statistic=False):
    print("Start to convert UAI data.")
    os.makedirs(r"./data/converted/UAI", exist_ok=True)
    pool = mp.Pool(mp.cpu_count())
    results = []
    with tqdm(total=len(uai_paper_path), desc="UAI Conversion", position=1) as pbar:
        for result in pool.imap(process_uai_data, zip(uai_paper_path, uai_review_path)):
            results.append(result)
            pbar.update()

    pool.close()
    pool.join()

    for index, r in enumerate(results):
        r = correct_double_periods(r)
        with open(fr"./data/converted/UAI/{index:03}.json", 'w') as fp:
            json.dump(r, fp)

    if statistic:
        path_list = [fr"./data/converted/UAI/{index:03}.json" for index in range(len(results))]
        years = []
        for r in results:
            years.append(r[2])
        statis = cal_statistic(years, path_list, range(2022, 2024))
        return statis

def datasetsLoad(test_num):
    iclr_data_path = glob.glob(r"./data/converted/ICLR/**.json")
    iclr_data_path.sort()
    nips_data_path = glob.glob(r"./data/converted/NeurIPS/**.json")
    nips_data_path.sort()
    uai_data_path = glob.glob(r"./data/converted/UAI/**.json")
    uai_data_path.sort()

    num_iclr = len(iclr_data_path)
    num_nips = len(nips_data_path)
    num_uai = len(uai_data_path)

    if test_num >= num_iclr:
        raise Exception("number for test is too large")
    test_idx = random.sample(range(num_iclr), test_num)
    train_iclr = []
    test_iclr = []
    for idx, i in enumerate(iclr_data_path):
        if idx in test_idx:
            test_iclr.append(i)
        else:
            train_iclr.append(i)

    print(f"Datasets loaded as below:")
    print(f"ICLR: {num_iclr}")
    print(f"NeurIPS: {num_nips}")
    print(f"UAI: {num_uai}")
    print(f"Using {test_num} samples from ICLR dataset to test.")
    print("-"*os.get_terminal_size().columns)
    return test_iclr, train_iclr + nips_data_path + uai_data_path

def save_dataset_chunks(train_datasets, chunk_size):
    shutil.rmtree("./datasets/reviewmt_train", ignore_errors=True)
    os.makedirs("./datasets/reviewmt_train", exist_ok=True)
    
    split_list = [train_datasets[i:i + chunk_size] for i in range(0, len(train_datasets), chunk_size)]
    
    for index, split in enumerate(split_list):
        with open(fr"./datasets/reviewmt_train/{index:02}.json", 'w') as fp:
            json.dump(split, fp)

def main():
    parser = argparse.ArgumentParser(description="Convert raw data to datasets")
    parser.add_argument("--split", default=True, help="whether to split train & test datasets.")
    parser.add_argument("--num_of_test", default=100, help="size of test datasets (random extraction from ICLR2024)")
    parser.add_argument("--chunk_size", default=2000, help="the chunk size of train datasets split.")
    parser.add_argument("--statistic", default=False, help="Whether to compile statistics of datasets.")
    parser.add_argument("--shuffle", default=True, help="whether to shuffle the datasets.")
    args = parser.parse_args()

    statistic = args.statistic
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    nips_review_path, nips_paper_path, iclr_review_path, iclr_paper_path, uai_review_path, uai_paper_path = read_all_data()
    INPUT = input("Is prepared data correct? [[y]/n]")
    if INPUT.lower() == 'n':
        exit()
    
    print("Start to convert all datasets")

    with tqdm(desc="Conversion Script", total=3, leave=True) as tdq:
        if statistic:
            statis = convert_nips(nips_review_path, nips_paper_path, statistic=True)
            num = (os.get_terminal_size().columns-7)//2-1
            print("-"*num, "NeurIPS", "-"*num)
            for year in range(2021, 2024):
                print(f"{year}_papers: {statis[year]['papers']}")
                print(f"{year}_reviews: {statis[year]['reviews']}")
                print(f"{year}_tokens: {statis[year]['tokens']}")
            print("-"*num, "NeurIPS", "-"*num)
        else:
            statis = convert_nips(nips_review_path, nips_paper_path, statistic=False)
        tdq.update()
        if statistic:
            statis = convert_iclr(iclr_review_path, iclr_paper_path, statistic=True)
            num = (os.get_terminal_size().columns-4)//2-1
            print("-"*num, "ICLR", "-"*num)
            for year in range(2017, 2025):
                print(f"{year}_papers: {statis[year]['papers']}")
                print(f"{year}_reviews: {statis[year]['reviews']}")
                print(f"{year}_tokens: {statis[year]['tokens']}")
            print("-"*num, "ICLR", "-"*num)
        else:
            convert_iclr(iclr_review_path, iclr_paper_path, statistic=False)
        tdq.update()
        if statistic:
            statis = convert_uai(uai_review_path, uai_paper_path, statistic=True)
            num = (os.get_terminal_size().columns-3)//2-1
            print("-"*num, "UAI", "-"*num)
            for year in range(2022, 2024):
                print(f"{year}_papers: {statis[year]['papers']}")
                print(f"{year}_reviews: {statis[year]['reviews']}")
                print(f"{year}_tokens: {statis[year]['tokens']}")
            print("-"*num, "UAI", "-"*num)
        else:
            convert_uai(uai_review_path, uai_paper_path, statistic=False)
        tdq.update()
    print("Conversion successfully completed!")
    print()
    
    if args.split:
        print("Start to split train and test datasets.")
        test_datasets_paths, train_datasets_paths = datasetsLoad(args.num_of_test)
        test_datasets = []
        train_datasets = []
        for p in test_datasets_paths:
            with open(p, 'r') as fp:
                content = json.load(fp)
            test_datasets.append(content)
        for p in train_datasets_paths:
            with open(p, 'r') as fp:
                content = json.load(fp)
            train_datasets.append(content)
        if args.shuffle:
            random.shuffle(test_datasets)
            random.shuffle(train_datasets)
        test_datasets = [i[0] for i in test_datasets]
        train_datasets = [i[0] for i in train_datasets]    
        with open(r"./datasets/reviewmt_test.json", 'w') as fp:
            json.dump(test_datasets, fp)
        if args.chunk_size <= 0:
            raise Exception("chunk_size must be greater than zero.")
        save_dataset_chunks(train_datasets, args.chunk_size)
        print("All datasets have been split and save in './datasets'")
if __name__ == '__main__':
    main()
