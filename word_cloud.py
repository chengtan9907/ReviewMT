import openreview
import re
from collections import Counter
import os
from tqdm import tqdm
import json
import glob

os.environ["http_proxy"] = "http://wolfcave.myds.me:987"
os.environ["https_proxy"] = "http://wolfcave.myds.me:987"
os.environ["all_proxy"] = "http://wolfcave.myds.me:988"

heap = []

class statistic_words:
    def __init__(self):
        self.counter = Counter()

    def add_text(self, text):
        # 匹配单词和短语
        words = re.findall(r'\b\w+\b', text.lower())
        phrases = []
        
        # 遍历单词列表，查找短语
        i = 0
        while i < len(words):
            if i < len(words) - 1 and words[i + 1] == 'learning':
                phrases.append(f"{words[i]} {words[i + 1]}")
                i += 2  # 跳过下一个单词
            else:
                phrases.append(words[i])
                i += 1

        self.counter.update(phrases)  # 统计单词和短语

    def add_text_directlly(self, text):
        text = [i.lower() for i in text]
        self.counter.update(text)  # 统计单词和短语

    def top_n(self, n):
        return self.counter.most_common(n)

def deal_iclr():
    sw = []
    client = openreview.Client(baseurl='https://api.openreview.net')
    with tqdm(desc="Getting ICLR keywords", total=8) as tqd:
        for year in range(2017, 2024):
            submissions = client.get_all_notes(invitation=f'ICLR.cc/{year}/conference/-/submission', details='directReplies')
            for submission in submissions:
                keywords = submission.content['keywords']
                if keywords:
                    if isinstance(keywords, dict):
                        keywords = keywords['value']
                    # sw.add_text_directlly(keywords)
                    sw.append(keywords)
                else:
                    keywords = submission.content['title']
                    # sw.add_text(keywords)
                    # sw.append(keywords)
            tqd.update()
        client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
        submissions = client.get_all_notes(invitation='ICLR.cc/2024/Conference/-/Submission', details='directReplies')
        for submission in submissions:
            keywords = submission.content['keywords']
            if keywords:
                if isinstance(keywords, dict):
                    keywords = keywords['value']
                # sw.add_text_directlly(keywords)
                sw.append(keywords)
            else:
                keywords = submission.content['title']
                # sw.add_text(keywords)
                sw.append(keywords)
        tqd.update()
    return sw

def deal_nips(sw, year):
    nips_path = glob.glob(fr"data/NeurIPS/{year}**/**/**.json")
    nips_path.sort()
    for i in nips_path:
        if 'content' in i:
            continue
        with open(i, 'r') as fp:
            content = json.load(fp)
        if isinstance(content, list):
            content = content[0]

        if not 'title' in content:
            continue

        if isinstance(content['title'], dict):
            title = content['title']['value']
        else:
            title = content['title']
        title = content['title']

        if isinstance(title, dict):
            title = title['value']

        sw.append(title)
        # sw.add_text(title)
    return sw

if __name__ == "__main__":
    sw = deal_iclr()
    with open("iclr_word_cloud.txt", 'w') as fp:
        for i in sw:
            fp.write(f"{i}\n")
    for year in range(2021, 2024):
        sw = []
        sw = deal_nips(sw, year)
        # result = sw.top_n(2000)
        # print_result = ""
        # for r in result:
        #     word = r[0].capitalize()
        #     word = word.replace(" ", "~")
        #     print_result += f"{word} "
        # print_result = print_result[:-1]
        with open(f"result_{year}_raw.txt", 'w') as fp:
            for i in sw:
                fp.write(f"{i}\n")