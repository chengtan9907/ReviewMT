import glob
import multiprocessing as mp

def find_name(raw_list, name):
    for i in raw_list:
        if name in i:
            return raw_list.index(i)
    return False
    
def read_all_data():
    # read NIPS data
    raw_paper_path = glob.glob(r"./data/tmp/NeurIPS/**/**.md")
    raw_paper_path.sort()
    raw_review_path = glob.glob(r"./data/NeurIPS/**/**/**.json")
    raw_review_path = [i for i in raw_review_path if not 'content' in i]
    nips_review_path = []
    nips_paper_path = []
    for paper in raw_paper_path:
        name = paper.split("/")[-1].replace(".md", "")
        index = find_name(raw_review_path, name)
        if index != False:
            nips_review_path.append(raw_review_path[index])
            nips_paper_path.append(paper)
    nips_review_path.sort()
    nips_paper_path.sort()
    print(f"nips_review: {len(nips_review_path)}")
    print(f"nips_paper: {len(nips_paper_path)}")

    # read ICLR data
    with open("./invalidPDF.txt", 'r') as fp:
        invalid_pdf = fp.readlines()
    invalid_pdf = [i.strip().replace("pdf", "md") for i in invalid_pdf]
    invalid_review = []
    for i in invalid_pdf:
        invalid_review.append(i.replace("md", "json"))

    raw_paper_path = glob.glob(r"./data/tmp/ICLR/**/**.md")
    raw_paper_path = [i for i in raw_paper_path if not i.split("/")[-1] in invalid_pdf]
    raw_review_path = glob.glob(r"./data/iclr_reviews/**.json")
    raw_review_path = [i for i in raw_review_path if not i.split("/")[-1] in invalid_review]
    iclr_review_path = []
    iclr_paper_path = []
    for paper in raw_paper_path:
        name = paper.split("/")[-1].replace(".md", "").replace("paper", "review")
        index = find_name(raw_review_path, name)
        if index != False:
            iclr_review_path.append(raw_review_path[index])
            iclr_paper_path.append(paper)
    iclr_review_path.sort()
    iclr_paper_path.sort()
    print(f"iclr_review: {len(iclr_review_path)}")
    print(f"iclr_paper: {len(iclr_paper_path)}")

    # read UAI data
    uai_review_path = glob.glob(r"./data/UAI/**/**/**.json")
    uai_review_path = [i for i in uai_review_path if not "_content" in i]
    uai_paper_path = glob.glob(r"./data/tmp/UAI/**/**.md")
    uai_review_path.sort()
    uai_paper_path.sort()
    print(f"uai_review: {len(uai_review_path)}")
    print(f"uai_paper: {len(uai_paper_path)}")

    return nips_review_path, nips_paper_path, iclr_review_path, iclr_paper_path, uai_review_path, uai_paper_path