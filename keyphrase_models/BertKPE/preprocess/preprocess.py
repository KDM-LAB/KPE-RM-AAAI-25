import os
import re
import sys
import json
import time
import torch
import codecs
import pickle
# import logging # Commenting all logging functions
import argparse
import unicodedata
import numpy as np
from tqdm import tqdm
import keyphrase_models.BertKPE.preprocess.prepro_utils

dataset_dict = {
    "openkp": [("EvalPublic", "eval"), ("Dev", "dev"), ("Train", "train")],
    # By Harsh Sharma
    # "kp20k": [("testing", "eval"), ("validation", "dev"), ("training", "train")],
    "kp20k": [("testing", "eval")],
}

# logger = logging.getLogger()
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# config param
dataset_class = "kp20k"
source_dataset_dir = "..\data\dataset"
output_path = "..\data\prepro_dataset"
max_src_seq_length = 300
min_src_seq_length = 20
max_trg_seq_length = 6
min_trg_seq_length = 0

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# original dataset loader
def openkp_loader(mode, source_dataset_dir):
    """ load source OpenKP dataset :'url', 'VDOM', 'text', 'KeyPhrases' """

    # logger.info("start loading %s data ..." % mode)
    source_path = os.path.join(source_dataset_dir, "OpenKP%s.jsonl" % mode)
    data_pairs = []
    with codecs.open(source_path, "r", "utf-8") as corpus_file:
        for idx, line in enumerate(tqdm(corpus_file)):
            json_ = json.loads(line)
            data_pairs.append(json_)
    return data_pairs


def kp20k_loader(
    mode,
    source_dataset_dir,
    src_fields=["title", "abstract"],
    trg_delimiter=";",
):

    """load source Kp20k dataset :'title', 'abstract', 'keyword'
    return : tuple : src_string, trg_string"""

    # logger.info("start loading %s data ..." % mode)
    source_path = os.path.join(source_dataset_dir, "kp20k_%s.json" % mode)

    data_pairs = []
    with codecs.open(source_path, "r", "utf-8") as corpus_file:
        for idx, line in enumerate(corpus_file):
            json_ = json.loads(line)

            src_str = ".".join([json_[f] for f in src_fields])
            data_pairs.append((src_str))
    return data_pairs


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# first stage preprocess
def openkp_refactor(examples, mode):
    # logger.info("strat refactor openkp %s data ..." % mode)

    have_phrase = True
    if mode == "EvalPublic":
        have_phrase = False

    return_pairs = []
    for idx, ex in enumerate(tqdm(examples)):
        doc_words, word2block, block_features = prepro_utils.refactor_text_vdom(
            text=ex["text"], VDOM=ex["VDOM"]
        )
        data = {}
        data["url"] = ex["url"]
        data["doc_words"] = doc_words
        data["word2block"] = word2block
        data["block_features"] = block_features
        if have_phrase:
            keyphrases = prepro_utils.clean_phrase(ex["KeyPhrases"])
            data["keyphrases"] = keyphrases

        return_pairs.append(data)
    return return_pairs

def kp20k_refactor(src_trgs_pairs, mode, valid_check=True):
    # logger.info("start refactor kp20k %s data ..." % mode)

    def tokenize_fn(text):
        """
        The tokenizer used in Meng et al. ACL 2017
        parse the feed-in text, filtering and tokenization
        keep [_<>,\(\)\.\'%], replace digits to 'DIGIT', split by [^a-zA-Z0-9_<>,\(\)\.\'%]
        :return: a list of tokens
        """
        DIGIT = "DIGIT"
        # remove line breakers
        text = re.sub(r"[\r\n\t]", " ", text)
        # pad spaces to the left and right of special punctuations
        text = re.sub(r"[_<>,\(\)\.\'%]", " \g<0> ", text)
        # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
        tokens = filter(
            lambda w: len(w) > 0, re.split(r"[^a-zA-Z0-9_<>,#&\+\*\(\)\.\'%]", text)
        )

        # replace the digit terms with DIGIT
        tokens = [w if not re.match("^\d+$", w) else DIGIT for w in tokens]

        return tokens

    # ---------------------------------------------------------------------------------------
    return_pairs = []
    for idx, (src) in enumerate(src_trgs_pairs):
        src_filter_flag = False

        src_tokens = tokenize_fn(src)

        return_pairs.append({"doc_words": src_tokens})

    return return_pairs


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# filter absent keyphrases
def filter_openkp_absent(examples):
    # logger.info("strat filter absent keyphrases ...")
    data_list = []
    null_urls, absent_urls = [], []
    for idx, ex in enumerate(tqdm(examples)):

        lower_words = [t.lower() for t in ex["doc_words"]]
        present_phrases = prepro_utils.find_answer(
            document=lower_words, answers=ex["keyphrases"]
        )
        if present_phrases is None:
            null_urls.append(ex["url"])
            continue
        if len(present_phrases["keyphrases"]) != len(ex["keyphrases"]):
            absent_urls.append(ex["url"])

        data = {}
        data["url"] = ex["url"]
        data["doc_words"] = ex["doc_words"]
        data["word2block"] = ex["word2block"]
        data["block_features"] = ex["block_features"]
        data["keyphrases"] = ex["keyphrases"]
        # -------------------------------------------------------
        # new added
        data["start_end_pos"] = present_phrases["start_end_pos"]
        data["present_keyphrases"] = present_phrases["keyphrases"]
        data_list.append(data)

    # logger.info("Null : number = {} , URL = {} ".format(len(null_urls), null_urls))
    # logger.info(
    #     "Absent : number = {} , URL = {} ".format(len(absent_urls), absent_urls)
    # )
    return data_list

def filter_kp20k_absent(examples):
    # logger.info("strat filter absent keyphrases for KP20k...")
    data_list = []

    null_ids, absent_ids = 0, 0

    url = 0
    for idx, ex in enumerate(examples):

        lower_words = [t.lower() for t in ex["doc_words"]]

        data = {}
        data["url"] = url
        data["doc_words"] = ex["doc_words"]
        data_list.append(data)
        url += 1

    # logger.info("Null : number = {} ".format(null_ids))
    # logger.info("Absent : number = {} ".format(absent_ids))
    return data_list


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# saver
def save_ground_truths(examples, filename, kp_key):
    with open(filename, "w", encoding="utf-8") as f_pred:
        for ex in tqdm(examples):
            data = {}
            data["url"] = ex["url"]
            data["KeyPhrases"] = ex[kp_key]
            f_pred.write("{}\n".format(json.dumps(data)))
        f_pred.close()
    # logger.info("Success save reference to %s" % filename)


def save_preprocess_data(data_list, filename):
    with open(filename, "w", encoding="utf-8") as fo:
        for data in tqdm(data_list):
            fo.write("{}\n".format(json.dumps(data)))
        fo.close()
    # logger.info("Success save file to %s \n" % filename)


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# main function
def main_preprocess(text, input_mode, save_mode):
    source_data = text.copy()

    # refactor source data
    refactor_data = data_refactor[dataset_class](source_data, input_mode)
    # logger.info("success refactor %s source data !" % input_mode)

    # filter absent keyphrases (kp20k use stem)
    if dataset_class == "openkp" and input_mode == "EvalPublic":
        feed_data = refactor_data
    else:
        feed_data = absent_filter[dataset_class](refactor_data)

    return feed_data

data_refactor = {"openkp": openkp_refactor, "kp20k": kp20k_refactor}
absent_filter = {"openkp": filter_openkp_absent, "kp20k": filter_kp20k_absent}

t0 = time.time()

def preprocess_gold(text: dict):
    """
    text: string containing title concatenated with abstract. (end title with period)
    """
    mode_dir = dataset_dict[dataset_class]

    return main_preprocess([text], mode_dir[0][0], mode_dir[0][1])
