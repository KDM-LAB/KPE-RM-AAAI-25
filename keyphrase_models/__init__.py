from keyphrase_models.keybert_model import keybert_keywords
from keyphrase_models.keybert_multiword_model import keybert_multiword_keywords
from keyphrase_models.keybert_30_model import keybert_30_keywords
from keyphrase_models.keybart_model import keybart_keywords
from keyphrase_models.patternrank_model import patternrank_keywords
from keyphrase_models.one2set_model import one2set_keywords
from keyphrase_models.bertkpe_model import bertkpe_keywords

from keyphrase_models.TFIDF import TFIDF_keywords
from keyphrase_models.YAKE import YAKE_keywords
from keyphrase_models.TextRank import TextRank_keywords
from keyphrase_models.SingleRank import SingleRank_keywords
from keyphrase_models.TopicRank import TopicRank_keywords
from keyphrase_models.PositionRank import PositionRank_keywords
from keyphrase_models.MultipartiteRank import MultipartiteRank_keywords
from keyphrase_models.KPMiner import KPMiner_keywords


model_dict = {"keybert":keybert_keywords,
            "keybert_multiword":keybert_multiword_keywords,
            "keybert_30":keybert_30_keywords,
            "keybart":keybart_keywords,
            "bertkpe":bertkpe_keywords,
            "patternrank":patternrank_keywords,
            "one2set":one2set_keywords,
            "tfidf":TFIDF_keywords,
            "yake":YAKE_keywords,
            "textrank":TextRank_keywords,
            "singlerank":SingleRank_keywords,
            "topicrank":TopicRank_keywords,
            "positionrank":PositionRank_keywords,
            "multipartiterank":MultipartiteRank_keywords,
            "kpminer":KPMiner_keywords,
            "promptrank":None}