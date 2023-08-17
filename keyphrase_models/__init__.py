from keyphrase_models.keybert_model import keybert_keywords
from keyphrase_models.keybart_model import keybart_keywords

from keyphrase_models.TFIDF import TFIDF_keywords
from keyphrase_models.YAKE import YAKE_keywords
from keyphrase_models.TextRank import TextRank_keywords
from keyphrase_models.SingleRank import SingleRank_keywords
from keyphrase_models.TopicRank import TopicRank_keywords
from keyphrase_models.PositionRank import PositionRank_keywords
from keyphrase_models.MultipartiteRank import MultipartiteRank_keywords
from keyphrase_models.KPMiner import KPMiner_keywords

from keyphrase_models.bertkpe_model import bertkpe_keywords

model_dict = {"keybert":keybert_keywords,
            "keybart":keybart_keywords,
            "tfidf":TFIDF_keywords,
            "yake":YAKE_keywords,
            "textrank":TextRank_keywords,
            "singlerank":SingleRank_keywords,
            "topicrank":TopicRank_keywords,
            "positionrank":PositionRank_keywords,
            "multipartiterank":MultipartiteRank_keywords,
            "kpminer":KPMiner_keywords,
            "bertkpe":bertkpe_keywords}