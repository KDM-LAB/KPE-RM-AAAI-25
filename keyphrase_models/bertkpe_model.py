from keyphrase_models.BertKPE.preprocess.preprocess import preprocess_gold
from keyphrase_models.BertKPE.scripts.test import testing_gold

def bertkpe_keywords(title: str="", abstract: str="", pdf_text: str="") -> str:
    if len(pdf_text) == 0:
        if len(title) != 0:
            if title[-1] == '.':
                text = title + " " + abstract
            else:
                text = title + ". " + abstract
        else:
            text = abstract
    else:
        text = pdf_text

    out = preprocess_gold(text)
    out = testing_gold(out)

    result = ";".join(" ".join(keyphrase) for keyphrase in out)

    return result