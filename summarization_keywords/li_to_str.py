import os
import json

# files = os.listdir()
# files.remove("li_to_str.py")
files = ["factorsum_keybert_keywords.json", "factorsum_patternrank_keywords.json", "factorsum_promptrank_keywords.json"]

for fil in files:
    print(fil)
    with open(fil, "r", encoding="utf-8") as f:
        keywords_dict = json.load(f)

    for k in keywords_dict.keys():
        if isinstance(keywords_dict[k]["ta_keywords"], list):
            keywords_dict[k]["ta_keywords"] = ";".join(keywords_dict[k]["ta_keywords"])

        if isinstance(keywords_dict[k]["pdf_keywords"], list):
            keywords_dict[k]["pdf_keywords"] = ";".join(keywords_dict[k]["pdf_keywords"])

    with open(fil, "w") as f:
        json.dump(keywords_dict, f)