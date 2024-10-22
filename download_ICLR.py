from tqdm import tqdm
import importlib
import fitz
import os

def isValidPDF_pathfile(pathfile):
    try:
        doc = fitz.open(pathfile)
        if doc.page_count < 1:
            return False
        return True
    except:
        return False

if __name__ == '__main__':
    years = [i for i in range(2017,2025)]
    print("Start to download ICLR raw data")
    outpath1 = r"data/iclr_papers"
    outpath2 = r"data/iclr_reviews"
    os.makedirs(outpath1, exist_ok=True)
    os.makedirs(outpath2, exist_ok=True)
    for year in tqdm(years, position=0, desc = "Processing:", leave=False):
        webcrawler = importlib.import_module(f"iclr.ICLR{year}")
        webcrawler.get_paper_and_review(outpath1, outpath2)

    print("All ICLR raw data have been downloaded.")