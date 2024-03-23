# from keyphrase_models.keybert_model import keybert_keywords
# from keyphrase_models.keybert_multiword_model import keybert_multiword_keywords
# from keyphrase_models.keybert_30_model import keybert_30_keywords
# from keyphrase_models.keybart_model import keybart_keywords
# from keyphrase_models.patternrank_model import patternrank_keywords
# from keyphrase_models.one2set_model import one2set_keywords
# from keyphrase_models.bertkpe_model import bertkpe_keywords
from keyphrase_models.promptrank_model import promptrank_keywords # Comment this line if you 
# wish to populate promptrank keyphrases using precomputed keyphrases

# from keyphrase_models.TFIDF import TFIDF_keywords
# from keyphrase_models.YAKE import YAKE_keywords
# from keyphrase_models.TextRank import TextRank_keywords
# from keyphrase_models.SingleRank import SingleRank_keywords
# from keyphrase_models.TopicRank import TopicRank_keywords
# from keyphrase_models.PositionRank import PositionRank_keywords
# from keyphrase_models.MultipartiteRank import MultipartiteRank_keywords
# from keyphrase_models.KPMiner import KPMiner_keywords

# Comment this dict if you wish to populate promptrank keyphrases using precomputed keyphrases
# model_dict = {"keybert":keybert_keywords,
#             "keybert_multiword":keybert_multiword_keywords,
#             "keybert_30":keybert_30_keywords,
#             "keybart":keybart_keywords,
#             "bertkpe":bertkpe_keywords,
#             "patternrank":patternrank_keywords,
#             "one2set":one2set_keywords,
#             "promptrank":promptrank_keywords,
#             "tfidf":TFIDF_keywords,
#             "yake":YAKE_keywords,
#             "textrank":TextRank_keywords,
#             "singlerank":SingleRank_keywords,
#             "topicrank":TopicRank_keywords,
#             "positionrank":PositionRank_keywords,
#             "multipartiterank":MultipartiteRank_keywords,
#             "kpminer":KPMiner_keywords}

# Uncomment this dict if you wish to populate promptrank keyphrases using precomputed keyphrases
# model_dict = {"keybert":keybert_keywords,
#             "keybert_multiword":keybert_multiword_keywords,
#             "keybert_30":keybert_30_keywords,
#             "keybart":keybart_keywords,
#             "bertkpe":bertkpe_keywords,
#             "patternrank":patternrank_keywords,
#             "one2set":one2set_keywords,
#             "promptrank":None,
#             "tfidf":TFIDF_keywords,
#             "yake":YAKE_keywords,
#             "textrank":TextRank_keywords,
#             "singlerank":SingleRank_keywords,
#             "topicrank":TopicRank_keywords,
#             "positionrank":PositionRank_keywords,
#             "multipartiterank":MultipartiteRank_keywords,
#             "kpminer":KPMiner_keywords}

model_summ_dict = {"keybert_longt5_summ":None,
            "patternrank_longt5_summ":None,
            "promptrank_longt5_summ":None,
            "keybert_pegasusx_summ":None,
            "patternrank_pegasusx_summ":None,
            "promptrank_pegasusx_summ":None,
            "keybert_memsum_summ":None,
            "patternrank_memsum_summ":None,
            "promptrank_memsum_summ":None,
            "keybert_factorsum_summ":None,
            "patternrank_factorsum_summ":None,
            "promptrank_factorsum_summ":None}

# model_dict.update(model_summ_dict)

model_dict = {"promptrank":promptrank_keywords}