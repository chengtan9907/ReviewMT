import json
import os
import torch.multiprocessing as mp
from tqdm import tqdm
import argparse
from src.module.statistic_module import cal_statistic
from src.module.readData import read_all_data
from src.module.convert_module import process_uai_data, process_iclr_data, process_nips_data, save_file
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    futures = []
    with ThreadPoolExecutor(max_workers = 100) as executor:
        for index, (meta, _, _) in enumerate(results):
            future = executor.submit(save_file, (meta, index))
            futures.append(future)
    as_completed(futures)

    for index, (meta, _, _) in enumerate(results):
        with open(fr"./data/converted/NeurIPS/{index:04}.json", 'w') as fp:
            json.dump(meta, fp)

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
    futures = []
    with ThreadPoolExecutor(max_workers = 100) as executor:
        for index, (meta, _, _) in enumerate(results):
            future = executor.submit(save_file, (meta, index))
            futures.append(future)
    as_completed(futures)
    
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
    futures = []
    with ThreadPoolExecutor(max_workers = 100) as executor:
        for index, (meta, _, _) in enumerate(results):
            future = executor.submit(save_file, (meta, index))
            futures.append(future)
    as_completed(futures)
    if statistic:
        path_list = [fr"./data/converted/UAI/{index:03}.json" for index in range(len(results))]
        years = []
        for r in results:
            years.append(r[2])
        statis = cal_statistic(years, path_list, range(2022, 2024))
    return statis

def main():

    parser = argparse.ArgumentParser(description="Convert raw data to datasets")
    parser.add_argument("--statistic", default=False, help="Whether to compile statistics of datasets.")
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
            print("----------NeurIPS----------")
            for year in range(2021, 2024):
                print(f"{year}_papers: {statis[year]['papers']}")
                print(f"{year}_reviews: {statis[year]['reviews']}")
                print(f"{year}_tokens: {statis[year]['tokens']}")
            print("----------NeurIPS----------")
        else:
            statis = convert_nips(nips_review_path, nips_paper_path, statistic=False)
        tdq.update()
        if statistic:
            statis = convert_iclr(iclr_review_path, iclr_paper_path, statistic=True)
            print("----------ICLR----------")
            for year in range(2017, 2025):
                print(f"{year}_papers: {statis[year]['papers']}")
                print(f"{year}_reviews: {statis[year]['reviews']}")
                print(f"{year}_tokens: {statis[year]['tokens']}")
            print("----------ICLR----------")
        else:
            convert_iclr(iclr_review_path, iclr_paper_path, statistic=False)
        tdq.update()
        if statistic:
            statis = convert_uai(uai_review_path, uai_paper_path, statistic=True)
            print("----------UAI----------")
            for year in range(2022, 2024):
                print(f"{year}_papers: {statis[year]['papers']}")
                print(f"{year}_reviews: {statis[year]['reviews']}")
                print(f"{year}_tokens: {statis[year]['tokens']}")
            print("----------UAI----------")
        else:
            convert_uai(uai_review_path, uai_paper_path, statistic=False)
        tdq.update()
    print("Conversion successfully completed!")

if __name__ == '__main__':
    main()
