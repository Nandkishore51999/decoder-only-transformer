# from datasets import load_dataset
#
# dataset = load_dataset("Squish42/bluemoon-fandom-1-1-rp-cleaned")
# a = 5



import json
json_data = json.load(open(r"C:\Workspace-ML\text_data\Bluemoon\bluemoon.train.json"))

text = ""
for i, each_conv in enumerate(json_data):

    conv = each_conv['conversations']
    for ele in conv:
        text += "\n" + ele["value"]

    if len(text) > 10_000_0000 or i==len(json_data)-1:
        with open(f"{each_conv['id']}.txt", "w") as file:
            file.write(text)
        text = ""