from keyphrase_models.keybert_model import keybert_keywords
from keyphrase_models.keybart_model import keybart_keywords

from keyphrase_models.TFIDF import TFIDF_keywords
from keyphrase_models.TextRank import TextRank_keywords
from keyphrase_models.KPMiner import KPMiner_keywords

model_dict = {"keybert":keybert_keywords,
            "keybart":keybart_keywords,
            "tfidf":TFIDF_keywords,
            "textrank":TextRank_keywords,
            "kpminer":KPMiner_keywords}