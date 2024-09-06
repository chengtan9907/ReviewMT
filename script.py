import json
import glob

l = glob.glob("./datasets/reviewmt_train/**.json")
l.sort()
for index, i in enumerate(l):
    with open(i, 'r') as fp:
        content = json.load(fp)
    print(f"{index:02}: {len(content)}")