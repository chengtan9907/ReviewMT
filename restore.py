import os
import shutil
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 定义文件复制的函数
def copy_file_1(file):
    name = file.split("/")[-1]
    shutil.copy(file, os.path.join(r"./data/tmp/PDFS/NeurIPS", name))

def copy_file_2(file):
    name = file.split("/")[-1]
    shutil.copy(file, os.path.join(r"./data/tmp/PDFS/iclr_papers", name))
    
def copy_file_3(file):
    name = file.split("/")[-1]
    shutil.copy(file, os.path.join(r"./data/tmp/PDFS/UAI", name))

# 获取所有 PDF 文件的路径
src = glob.glob(r"./RawData/data/NeurIPS/**/**/**.pdf")
src2 = glob.glob(r"./RawData/data/iclr_papers/**.pdf")
src3 = glob.glob(r"./RawData/data/UAI/**/**/**.pdf")


# 使用多线程进行文件复制
with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(copy_file_1, src), total=len(src)))
    list(tqdm(executor.map(copy_file_2, src2), total=len(src2)))
    list(tqdm(executor.map(copy_file_3 , src3), total=len(src3)))