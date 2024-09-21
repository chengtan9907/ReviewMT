import glob
import multiprocessing as mp

def find_name(raw_list, name):
    for i in raw_list:
        if name in i:
            return raw_list.index(i)
    return False

def process_dataset(raw_paper_path, raw_review_path, invalid_papers=[], invalid_reviews=[]):
    paper_paths = [i for i in raw_paper_path if i.split("/")[-1] not in invalid_papers]
    review_paths = [i for i in raw_review_path if i.split("/")[-1] not in invalid_reviews]
    
    matched_review_path = []
    matched_paper_path = []
    
    for paper in paper_paths:
        name = paper.split("/")[-1].replace(".md", "")
        if 'paper' in name:
            name = name.replace("paper", "review")
        index = find_name(review_paths, name)
        if index != False:
            matched_review_path.append(review_paths[index])
            matched_paper_path.append(paper)
    
    matched_review_path.sort()
    matched_paper_path.sort()
    
    return matched_review_path, matched_paper_path

def read_all_data():
    # Process NIPS dataset
    raw_paper_path_nips = glob.glob(r"./data/tmp/NeurIPS/**/**.md")
    raw_paper_path_nips.sort()
    raw_review_path_nips = glob.glob(r"./data/NeurIPS/**/**/**.json")
    raw_review_path_nips = [i for i in raw_review_path_nips if 'content' not in i]
    
    # Process ICLR dataset
    with open("./invalidPDF.txt", 'r') as fp:
        invalid_pdf = [i.strip().replace("pdf", "md") for i in fp.readlines()]
    invalid_review = [i.replace("md", "json") for i in invalid_pdf]
    
    raw_paper_path_iclr = glob.glob(r"./data/tmp/ICLR/**/**.md")
    raw_review_path_iclr = glob.glob(r"./data/iclr_reviews/**.json")

    # Process UAI dataset
    raw_paper_path_uai = glob.glob(r"./data/tmp/UAI/**/**.md")
    raw_review_path_uai = glob.glob(r"./data/UAI/**/**/**.json")
    raw_review_path_uai = [i for i in raw_review_path_uai if '_content' not in i]

    # Create a pool for multiprocessing
    with mp.Pool(processes=3) as pool:
        # Parallelize the dataset processing
        nips_data = pool.apply_async(process_dataset, (raw_paper_path_nips, raw_review_path_nips))
        iclr_data = pool.apply_async(process_dataset, (raw_paper_path_iclr, raw_review_path_iclr, invalid_pdf, invalid_review))
        uai_data = pool.apply_async(process_dataset, (raw_paper_path_uai, raw_review_path_uai))

        # Collect the results
        nips_review_path, nips_paper_path = nips_data.get()
        iclr_review_path, iclr_paper_path = iclr_data.get()
        uai_review_path, uai_paper_path = uai_data.get()

    print(f"nips_review: {len(nips_review_path)}")
    print(f"nips_paper: {len(nips_paper_path)}")
    print(f"iclr_review: {len(iclr_review_path)}")
    print(f"iclr_paper: {len(iclr_paper_path)}")
    print(f"uai_review: {len(uai_review_path)}")
    print(f"uai_paper: {len(uai_paper_path)}")

    return nips_review_path, nips_paper_path, iclr_review_path, iclr_paper_path, uai_review_path, uai_paper_path

if __name__ == "__main__":
    read_all_data()
