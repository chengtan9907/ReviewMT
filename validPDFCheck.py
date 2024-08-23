import PyPDF2
import glob
import os
import tqdm

invalid_pdf_path_list = []

def is_valid_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # 尝试获取文件信息以确保它是一个有效的PDF文件
            if reader.pages and len(reader.pages) > 0:
                return True
            else:
                return False
    except Exception as e:
        # 捕获异常并返回False，表示文件无效或损坏
        print(f"Error: {e}")
        invalid_pdf_path_list.append(file_path)
        return False


pdfFileList = os.listdir(r"./data/iclr_papers")
pdfFileList.sort()
validNum = 0
invalidNum = 0
for pdf in tqdm.tqdm(pdfFileList):
    pdfdir = os.path.join(r"./data/iclr_papers", pdf)
    if is_valid_pdf(pdfdir):
        validNum += 1
    else:
        invalidNum += 1
print(f"Valid_number: {validNum}\nInvalid_number: {invalidNum}\nTotal_number: {len(pdfFileList)}")
pdf2017 = glob.glob(r"./data/iclr_papers/ICLR_2017_paper_**.pdf")
pdf2018 = glob.glob(r"./data/iclr_papers/ICLR_2018_paper_**.pdf")
pdf2019 = glob.glob(r"./data/iclr_papers/ICLR_2019_paper_**.pdf")
pdf2020 = glob.glob(r"./data/iclr_papers/ICLR_2020_paper_**.pdf")
pdf2021 = glob.glob(r"./data/iclr_papers/ICLR_2021_paper_**.pdf")
pdf2022 = glob.glob(r"./data/iclr_papers/ICLR_2022_paper_**.pdf")
pdf2023 = glob.glob(r"./data/iclr_papers/ICLR_2023_paper_**.pdf")
pdf2024 = glob.glob(r"./data/iclr_papers/ICLR_2024_paper_**.pdf")
print(f"2017: {len(pdf2017)}")
print(f"2018: {len(pdf2018)}")
print(f"2019: {len(pdf2019)}")
print(f"2020: {len(pdf2020)}")
print(f"2021: {len(pdf2021)}")
print(f"2022: {len(pdf2022)}")
print(f"2023: {len(pdf2023)}")
print(f"2024: {len(pdf2024)}")

with open("./invalidPDF.txt", 'w') as fp:
    for line in invalid_pdf_path_list:
        fp.write(line)
        fp.write("\n")